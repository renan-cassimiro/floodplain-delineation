"""Stage 1: input validation and spatial preparation.

* Validates files, DEM metadata and the pilot-area geometry.
* Reprojects the pilot geometry to the DEM CRS (explicit; the DEM is never reprojected).
* Computes the processing extent (pilot area + context buffer) as an integer pixel
  window of the original DEM grid, so no resampling ever happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pyproj
import rasterio
import shapely
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .config import Config
from .exceptions import InputError
from .logging_utils import WarningCollector, get_logger
from .spatial import (
    buffer_distance_in_crs_units,
    check_supported_crs,
    linear_units_factor,
    to_pyproj,
)

log = get_logger("inputs")

_POLYGONAL = ("Polygon", "MultiPolygon")


@dataclass(frozen=True)
class DemInfo:
    path: Path
    crs: RioCRS
    pcrs: pyproj.CRS
    transform: Affine
    width: int
    height: int
    bounds: Tuple[float, float, float, float]
    nodata: float
    dtype: str
    driver: str

    @property
    def resolution(self) -> Tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)


@dataclass
class SpatialDomain:
    pilot_full: BaseGeometry          # dissolved pilot geometry, DEM CRS (may extend beyond DEM)
    pilot: BaseGeometry               # pilot ∩ DEM footprint (used for final clipping)
    source_crs: str                   # CRS of the pilot file (as written)
    n_features: int
    coverage_fraction: float          # area(pilot ∩ DEM) / area(pilot)
    window: Window                    # processing window on the DEM grid (integers)
    window_transform: Affine
    extent_bounds: Tuple[float, float, float, float]
    requested_bounds: Tuple[float, float, float, float]
    truncated_sides: List[str]
    buffer_info: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def validate_input_files(config: Config) -> None:
    """Check that the DEM and the pilot-area file exist and are readable."""
    for label, path in (("DEM", config.inputs.dem), ("Pilot-area GeoPackage", config.inputs.pilot_area)):
        if not path.exists():
            raise InputError(f"{label} file not found: {path}. Check 'inputs' in the configuration.")
        if not path.is_file():
            raise InputError(f"{label} path is not a file: {path}")
        try:
            with open(path, "rb"):
                pass
        except OSError as exc:
            raise InputError(f"{label} file is not readable: {path} ({exc})") from exc
    if config.inputs.pilot_area.suffix.lower() != ".gpkg":
        raise InputError(f"Pilot area must be a GeoPackage (.gpkg), got: {config.inputs.pilot_area.name}")


def read_dem_info(path: Path, warnings: Optional[WarningCollector] = None) -> DemInfo:
    """Read and validate DEM metadata."""
    try:
        src = rasterio.open(path)
    except Exception as exc:
        raise InputError(f"Cannot open DEM {path} as a raster: {exc}") from exc
    with src:
        if src.count != 1:
            raise InputError(f"DEM must be single-band, but {path.name} has {src.count} bands.")
        pcrs = check_supported_crs(src.crs, f"DEM {path.name}")
        if src.nodata is None:
            raise InputError(
                f"DEM {path.name} has no nodata value in its metadata. Set one explicitly "
                "(e.g. `gdal_edit.py -a_nodata -9999 dem.tif`) so nodata handling is unambiguous."
            )
        t = src.transform
        if abs(t.b) > 0 or abs(t.d) > 0 or t.a <= 0 or t.e >= 0:
            raise InputError(
                "Only north-up, non-rotated DEM grids are supported "
                f"(transform: {tuple(t)[:6]}). Warp the DEM to a north-up grid first."
            )
        if not (src.width > 1 and src.height > 1):
            raise InputError("DEM must have at least 2x2 cells.")
        if abs(abs(t.a) - abs(t.e)) > 1e-9 * abs(t.a) and warnings is not None:
            warnings.add(
                f"DEM cells are not square ({abs(t.a)} x {abs(t.e)} CRS units); "
                "area and length diagnostics use both dimensions."
            )
        return DemInfo(
            path=path, crs=src.crs, pcrs=pcrs, transform=t, width=src.width, height=src.height,
            bounds=tuple(src.bounds), nodata=float(src.nodata), dtype=src.dtypes[0], driver=src.driver,
        )


def _polygonal_part(geom: BaseGeometry) -> Optional[BaseGeometry]:
    """Keep only polygonal components of a geometry (None when nothing remains)."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in _POLYGONAL:
        return geom
    parts = [g for g in shapely.get_parts(geom) if g.geom_type in _POLYGONAL and not g.is_empty]
    if not parts:
        return None
    return shapely.union_all(parts)


def _repair(geom: BaseGeometry) -> Optional[BaseGeometry]:
    return _polygonal_part(shapely.make_valid(geom)) if not geom.is_valid else geom


def read_pilot_geometry(
    path: Path, layer: Optional[str], dem: DemInfo, warnings: WarningCollector
) -> Tuple[BaseGeometry, str, int]:
    """Read, validate, repair, reproject to the DEM CRS and dissolve the pilot area.

    Returns ``(geometry_in_dem_crs, source_crs_string, n_input_features)``.
    """
    try:
        import pyogrio

        layers = [str(name) for name, _ in pyogrio.list_layers(path)]
    except Exception as exc:
        raise InputError(f"Cannot read GeoPackage {path}: {exc}") from exc
    if layer is not None and layer not in layers:
        raise InputError(f"Layer '{layer}' not found in {path.name}. Available layers: {layers}.")
    if layer is None and len(layers) > 1:
        warnings.add(
            f"GeoPackage has several layers {layers}; using the first ('{layers[0]}'). "
            "Set inputs.pilot_layer to choose explicitly."
        )
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception as exc:
        raise InputError(f"Cannot read pilot area from {path}: {exc}") from exc

    n_input = len(gdf)
    if n_input == 0:
        raise InputError("The pilot-area layer contains no features.")
    if gdf.crs is None:
        raise InputError(
            "The pilot-area layer has no CRS. Assign one in your GIS before running; "
            "the workflow will not guess a coordinate system."
        )
    source_crs = gdf.crs.to_string()

    geoms = gdf.geometry
    keep = ~(geoms.isna() | geoms.is_empty)
    if not keep.all():
        warnings.add(f"Dropped {int((~keep).sum())} null/empty pilot-area feature(s).")
    geoms = geoms[keep]
    poly = geoms.geom_type.isin(_POLYGONAL)
    if not poly.all():
        warnings.add(f"Dropped {int((~poly).sum())} non-polygon feature(s) from the pilot-area layer.")
    geoms = geoms[poly]
    if len(geoms) == 0:
        raise InputError("The pilot-area layer contains no valid polygon geometries.")

    repaired, kept = 0, []
    for g in geoms:
        if not g.is_valid:
            repaired += 1
            g = _repair(g)
        if g is not None and not g.is_empty:
            kept.append(g)
    if repaired:
        warnings.add(f"Repaired {repaired} invalid pilot-area polygon(s) with make_valid.")
    if not kept:
        raise InputError("All pilot-area geometries are empty after validity repair.")

    series = gpd.GeoSeries(kept, crs=gdf.crs)
    if pyproj.CRS(series.crs) != dem.pcrs:
        log.info("Reprojecting pilot area from %s to the DEM CRS (%s).", source_crs, dem.crs.to_string())
        try:
            series = series.to_crs(dem.pcrs)
        except Exception as exc:
            raise InputError(f"Cannot transform pilot area to the DEM CRS: {exc}") from exc
    if not all(math.isfinite(v) for g in series for v in g.bounds):
        raise InputError("Pilot-area geometry has non-finite coordinates after reprojection.")

    fixed = []
    for g in series:
        g = _repair(g)
        if g is not None:
            fixed.append(g)
    merged = _polygonal_part(shapely.union_all(fixed))
    if merged is None or merged.area <= 0:
        raise InputError("Pilot-area geometry is empty or has zero area.")
    return merged, source_crs, n_input


def _buffer_geometry(geom: BaseGeometry, dem: DemInfo, distance: float, units: str) -> Tuple[BaseGeometry, Dict[str, object]]:
    """Buffer ``geom`` (DEM CRS). Geographic DEMs are buffered in a local UTM CRS."""
    info: Dict[str, object] = {"requested_distance": distance, "units": units}
    if distance == 0:
        info["method"] = "none"
        return geom, info
    direct = buffer_distance_in_crs_units(distance, units, dem.pcrs, dem.transform)
    if direct is not None:
        info.update(method="dem_crs", distance_in_crs_units=direct, buffer_crs=dem.crs.to_string())
        return geom.buffer(direct), info
    series = gpd.GeoSeries([geom], crs=dem.pcrs)
    utm = series.estimate_utm_crs()
    buffered = series.to_crs(utm).buffer(distance).to_crs(dem.pcrs).iloc[0]
    info.update(method="projected_crs", buffer_crs=utm.to_string())
    log.info("Geographic DEM CRS: buffer computed in %s and transformed back.", utm.to_string())
    return buffered, info


def bounds_to_window(
    bounds: Tuple[float, float, float, float], dem: DemInfo
) -> Tuple[Window, Tuple[float, float, float, float], List[str]]:
    """Smallest integer window of the DEM grid covering ``bounds`` (clipped to the DEM).

    Returns ``(window, actual_bounds, truncated_sides)``; a side is truncated when the DEM
    edge cuts the requested extent.
    """
    w = from_bounds(*bounds, transform=dem.transform)
    eps = 1e-6
    c0 = max(0, math.floor(w.col_off + eps))
    r0 = max(0, math.floor(w.row_off + eps))
    c1 = min(dem.width, math.ceil(w.col_off + w.width - eps))
    r1 = min(dem.height, math.ceil(w.row_off + w.height - eps))
    if c1 <= c0 or r1 <= r0:
        raise InputError("The processing extent does not overlap the DEM grid.")
    window = Window(c0, r0, c1 - c0, r1 - r0)
    actual = window_bounds(window, dem.transform)
    tol = 1e-6 * min(dem.resolution)
    sides = []
    if bounds[0] < actual[0] - tol:
        sides.append("west")
    if bounds[1] < actual[1] - tol:
        sides.append("south")
    if bounds[2] > actual[2] + tol:
        sides.append("east")
    if bounds[3] > actual[3] + tol:
        sides.append("north")
    return window, actual, sides


def prepare_spatial_domain(dem: DemInfo, config: Config, warnings: WarningCollector) -> SpatialDomain:
    """Build the pilot geometry, its DEM intersection and the processing window."""
    pilot_full, src_crs, n_features = read_pilot_geometry(
        config.inputs.pilot_area, config.inputs.pilot_layer, dem, warnings
    )
    footprint = box(*dem.bounds)
    inter = _polygonal_part(pilot_full.intersection(footprint))
    if inter is None or inter.area <= 0:
        raise InputError(
            "The pilot area does not intersect the DEM footprint. Check that both datasets cover "
            f"the same region. DEM bounds: {dem.bounds}; pilot bounds (DEM CRS): {pilot_full.bounds}."
        )
    coverage = inter.area / pilot_full.area
    if coverage < 1 - 1e-9:
        warnings.add(
            f"Only {coverage:.1%} of the pilot area lies within the DEM footprint; the final map is "
            "restricted to the covered part."
        )

    buffer_cfg = config.processing
    buffered, binfo = _buffer_geometry(pilot_full, dem, buffer_cfg.buffer_distance, buffer_cfg.buffer_units)
    if buffer_cfg.buffer_distance == 0:
        warnings.add(
            "buffer_distance is 0: hydrological processing runs on a tightly clipped DEM and is "
            "exposed to artificial edge effects."
        )
    requested = tuple(buffered.bounds)
    window, actual, truncated = bounds_to_window(requested, dem)
    if truncated:
        warnings.add(
            f"The requested context buffer is truncated by the DEM edge on the {', '.join(truncated)} "
            "side(s); flow routing near those sides lacks upstream context."
        )
    warnings_note = (
        "A buffer cannot guarantee correct upstream hydrology if the DEM does not extend "
        "sufficiently beyond the area of interest."
    )
    binfo["note"] = warnings_note
    return SpatialDomain(
        pilot_full=pilot_full, pilot=inter, source_crs=src_crs, n_features=n_features,
        coverage_fraction=coverage, window=window,
        window_transform=rasterio.windows.transform(window, dem.transform),
        extent_bounds=actual, requested_bounds=requested, truncated_sides=truncated, buffer_info=binfo,
    )
