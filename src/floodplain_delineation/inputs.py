"""Stage 1: input validation and spatial preparation (processing CRS and master grid).

* Validates files, DEM metadata and the pilot-area geometry.
* Resolves the processing CRS (projected, linear units) and the target resolution.
* Builds the master processing grid (CRS, transform, width, height): the one grid shared by every
  raster derived by the pipeline. The pilot geometry is reprojected to the processing CRS here.
* When the processing CRS and resolution equal the DEM's, the grid is a window of the DEM grid and no
  resampling happens; otherwise Stage 0 (:mod:`reprojection`) warps the DEM onto the master grid.
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
    ProcessingCrs,
    cell_geometry,
    check_supported_crs,
    densified_footprint,
    resolve_processing_crs,
    same_crs,
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


@dataclass(frozen=True)
class ProcessingGrid:
    """The master grid. Every raster derived by the pipeline has exactly this CRS/transform/shape."""

    crs: RioCRS
    pcrs: pyproj.CRS
    transform: Affine
    width: int
    height: int
    identical_to_source: bool                 # True: a window of the DEM grid (no resampling)
    source_window: Optional[Window] = None    # that window, when identical_to_source

    @property
    def res_x(self) -> float:
        return abs(self.transform.a)

    @property
    def res_y(self) -> float:
        return abs(self.transform.e)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)

    def describe(self) -> Dict[str, object]:
        return {
            "crs": self.crs.to_string(),
            "width": self.width,
            "height": self.height,
            "cells": self.width * self.height,
            "resolution": [self.res_x, self.res_y],
            "transform": list(tuple(self.transform))[:6],
            "bounds": list(self.bounds),
            "identical_to_source_grid": self.identical_to_source,
        }


@dataclass
class SpatialDomain:
    pilot_full: BaseGeometry          # dissolved pilot geometry, processing CRS (may extend beyond the DEM)
    pilot: BaseGeometry               # pilot ∩ DEM footprint (processing CRS; used for final clipping)
    source_crs: str                   # CRS of the pilot file (as written)
    n_features: int
    coverage_fraction: float          # area(pilot ∩ DEM footprint) / area(pilot)
    window: Optional[Window]          # processing window on the DEM grid; None when the grid is warped
    window_transform: Affine          # transform of the master grid
    extent_bounds: Tuple[float, float, float, float]      # master grid bounds (processing CRS)
    requested_bounds: Tuple[float, float, float, float]   # pilot (+ buffer) bounds before clipping
    truncated_sides: List[str]
    buffer_info: Dict[str, object] = field(default_factory=dict)
    grid: Optional[ProcessingGrid] = None
    processing_crs: Optional[ProcessingCrs] = None
    resolution_info: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def validate_input_files(config: Config) -> None:
    """Check that the DEM and (when given) the pilot-area file exist and are readable."""
    checks = [("DEM", config.inputs.dem)]
    if config.inputs.pilot_area is not None:
        checks.append(("Pilot-area GeoPackage", config.inputs.pilot_area))
    for label, path in checks:
        if not path.exists():
            raise InputError(f"{label} file not found: {path}. Check 'inputs' in the configuration.")
        if not path.is_file():
            raise InputError(f"{label} path is not a file: {path}")
        try:
            with open(path, "rb"):
                pass
        except OSError as exc:
            raise InputError(f"{label} file is not readable: {path} ({exc})") from exc
    if config.inputs.pilot_area is not None and config.inputs.pilot_area.suffix.lower() != ".gpkg":
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
    path: Path, layer: Optional[str], target_pcrs: pyproj.CRS, warnings: WarningCollector
) -> Tuple[BaseGeometry, str, int]:
    """Read, validate, repair, reproject to the processing CRS and dissolve the pilot area.

    Returns ``(geometry_in_processing_crs, source_crs_string, n_input_features)``.
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
    if not same_crs(pyproj.CRS(series.crs), target_pcrs):
        log.info("Reprojecting pilot area from %s to the processing CRS (%s).", source_crs, target_pcrs.name)
        try:
            series = series.to_crs(target_pcrs)
        except Exception as exc:
            raise InputError(f"Cannot transform pilot area to the processing CRS: {exc}") from exc
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


def _buffer_geometry(
    geom: BaseGeometry, proc: ProcessingCrs, res: Tuple[float, float], distance: float, units: str
) -> Tuple[BaseGeometry, Dict[str, object]]:
    """Buffer ``geom`` (processing CRS, projected). ``cells`` = processing-grid cells."""
    info: Dict[str, object] = {"requested_distance": distance, "units": units}
    if distance == 0:
        info["method"] = "none"
        return geom, info
    direct = distance * max(res) if units == "cells" else distance / proc.unit_factor
    info.update(method="processing_crs", distance_in_crs_units=direct, buffer_crs=proc.crs.to_string())
    return geom.buffer(direct), info


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

def resolve_target_resolution(
    dem: DemInfo, proc: ProcessingCrs, target: Optional[float], warnings: Optional[WarningCollector] = None
) -> Tuple[Tuple[float, float], Dict[str, object]]:
    """Resolution (x, y) of the master grid in processing-CRS units, and how it was obtained.

    * ``target`` given: that value (square cells);
    * ``null`` and the DEM already in the processing CRS: the DEM's own resolution (kept exactly);
    * ``null`` otherwise: the resolution GDAL suggests for the whole DEM (``calculate_default_transform``:
      square cells, preserving roughly the same number of cells along the DEM diagonal). Deterministic
      for a given DEM, CRS and GDAL version.
    """
    same = same_crs(dem.pcrs, proc.pcrs)
    if target is not None:
        res, how = (float(target), float(target)), "configured (processing.target_resolution)"
    elif same:
        res, how = dem.resolution, "native resolution of the DEM (already in the processing CRS)"
    else:
        from rasterio.warp import calculate_default_transform

        try:
            suggested, _, _ = calculate_default_transform(dem.crs, proc.crs, dem.width, dem.height, *dem.bounds)
        except Exception as exc:
            raise InputError(f"Cannot derive a target resolution for the processing CRS: {exc}") from exc
        r = abs(suggested.a)
        if not (math.isfinite(r) and r > 0):
            raise InputError("GDAL could not suggest a valid target resolution; set processing.target_resolution.")
        res, how = (r, r), "derived: GDAL calculate_default_transform over the full DEM (square cells)"
    src = cell_geometry(dem.pcrs, dem.transform, dem.bounds)
    src_size_m = math.sqrt(src.area_m2)
    tgt_size_m = math.sqrt(res[0] * res[1]) * proc.unit_factor
    info: Dict[str, object] = {
        "target_resolution_crs_units": list(res),
        "source": how,
        "source_cell_size_m_approx": src_size_m,
        "target_cell_size_m": tgt_size_m,
        "target_to_source_cell_size_ratio": tgt_size_m / src_size_m if src_size_m > 0 else None,
    }
    if warnings is not None and src_size_m > 0:
        ratio = tgt_size_m / src_size_m
        if ratio > 1.05:
            warnings.add(
                f"The processing resolution ({tgt_size_m:.2f} m) is coarser than the source DEM "
                f"(~{src_size_m:.2f} m): terrain detail is discarded by the reprojection."
            )
        elif ratio < 0.95:
            warnings.add(
                f"The processing resolution ({tgt_size_m:.2f} m) is finer than the source DEM "
                f"(~{src_size_m:.2f} m): no real detail is added, but the number of cells grows."
            )
    return res, info


def _snap_grid(
    bounds: Tuple[float, float, float, float], res: Tuple[float, float], pcrs: pyproj.CRS, crs: RioCRS
) -> ProcessingGrid:
    """Grid whose lattice is anchored at the CRS origin (multiples of the resolution) and covers ``bounds``."""
    eps = 1e-9
    rx, ry = res
    left, bottom, right, top = bounds
    c0, c1 = math.floor(left / rx + eps), math.ceil(right / rx - eps)
    r0, r1 = math.floor(bottom / ry + eps), math.ceil(top / ry - eps)
    width, height = int(c1 - c0), int(r1 - r0)
    if width < 1 or height < 1:
        raise InputError("The processing extent is smaller than one cell of the processing grid.")
    transform = Affine(rx, 0.0, c0 * rx, 0.0, -ry, r1 * ry)
    return ProcessingGrid(crs, pcrs, transform, width, height, identical_to_source=False)


def prepare_spatial_domain(
    dem: DemInfo,
    config: Config,
    warnings: WarningCollector,
) -> SpatialDomain:
    """Resolve the processing CRS/resolution, reproject the pilot area and build the master grid."""
    cfg = config.processing
    proc = resolve_processing_crs(cfg.crs)
    res, res_info = resolve_target_resolution(dem, proc, cfg.target_resolution, warnings)
    same = same_crs(dem.pcrs, proc.pcrs)
    identical = same and math.isclose(res[0], dem.resolution[0], rel_tol=1e-9) and math.isclose(
        res[1], dem.resolution[1], rel_tol=1e-9
    )
    res_info["grid_alignment"] = (
        "window of the source DEM grid (same CRS and resolution: no resampling)"
        if identical else "lattice anchored at the CRS origin (multiples of the resolution)"
    )

    # Footprint of the DEM in the processing CRS: edges densified, so curved/rotated footprints are not underestimated.
    footprint = densified_footprint(tuple(dem.bounds), dem.pcrs, proc.pcrs)
    fbounds = footprint.bounds

    def build_grid(requested: Tuple[float, float, float, float]):
        """Master grid for ``requested`` bounds, clipped to the DEM footprint; also the truncated sides."""
        clipped = (max(requested[0], fbounds[0]), max(requested[1], fbounds[1]),
                   min(requested[2], fbounds[2]), min(requested[3], fbounds[3]))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise InputError("The processing extent does not overlap the DEM grid.")
        tol = 1e-6 * min(res)
        sides = [n for n, cut in (("west", requested[0] < fbounds[0] - tol), ("south", requested[1] < fbounds[1] - tol),
                                  ("east", requested[2] > fbounds[2] + tol), ("north", requested[3] > fbounds[3] + tol)) if cut]
        if identical:
            window, actual, _ = bounds_to_window(clipped, dem)
            grid = ProcessingGrid(
                proc.crs, proc.pcrs, rasterio.windows.transform(window, dem.transform), int(window.width),
                int(window.height), identical_to_source=True, source_window=window,
            )
            return grid, window, sides
        return _snap_grid(clipped, res, proc.pcrs, proc.crs), None, sides

    # ------------------------------------------------------------------
    # No pilot area: the whole DEM footprint is the processing and final area.
    # ------------------------------------------------------------------
    if not config.inputs.pilot_area:
        warnings.add(
            "No pilot area was provided; the DEM footprint (bounding box in the processing CRS) is used "
            "as the processing and final output area."
        )
        if cfg.buffer_distance > 0:
            warnings.add(
                "buffer_distance is ignored because no pilot area was provided. The DEM footprint already "
                "defines the complete available processing extent, and no raster context exists outside the DEM."
            )
        grid, window, _ = build_grid(fbounds)
        pilot = box(*grid.bounds)
        warnings.add(
            "The processing area coincides with the DEM footprint. Hydrological routing near the DEM boundaries "
            "may lack upstream context and may be affected by artificial edge effects."
        )
        return SpatialDomain(
            pilot_full=pilot, pilot=pilot, source_crs=dem.crs.to_string(), n_features=1, coverage_fraction=1.0,
            window=window, window_transform=grid.transform, extent_bounds=grid.bounds, requested_bounds=fbounds,
            truncated_sides=["left", "right", "top", "bottom"],
            buffer_info={
                "distance": 0, "units": cfg.buffer_units, "applied": False,
                "reason": "No pilot area was provided; the DEM footprint defines the complete processing and output extent.",
                "note": "No external context is available beyond the DEM boundary. Flow routing near the DEM edges "
                        "may therefore be affected by artificial boundary effects.",
            },
            grid=grid, processing_crs=proc, resolution_info=res_info,
        )

    # ------------------------------------------------------------------
    # Normal workflow: read the pilot area and reproject it to the processing CRS.
    # ------------------------------------------------------------------
    pilot_full, src_crs, n_features = read_pilot_geometry(
        config.inputs.pilot_area, config.inputs.pilot_layer, proc.pcrs, warnings
    )
    inter = _polygonal_part(pilot_full.intersection(footprint))
    if inter is None or inter.area <= 0:
        raise InputError(
            "The pilot area does not intersect the DEM footprint. Check that both datasets cover the same region. "
            f"DEM footprint bounds (processing CRS): {tuple(round(v, 3) for v in fbounds)}; "
            f"pilot bounds (processing CRS): {pilot_full.bounds}."
        )
    coverage = inter.area / pilot_full.area
    if coverage < 1 - 1e-9:
        warnings.add(
            f"Only {coverage:.1%} of the pilot area lies within the DEM footprint; "
            "the final map is restricted to the covered part."
        )
    buffered, binfo = _buffer_geometry(pilot_full, proc, res, cfg.buffer_distance, cfg.buffer_units)
    if cfg.buffer_distance == 0:
        warnings.add(
            "buffer_distance is 0: hydrological processing runs on a tightly clipped DEM and is exposed to "
            "artificial edge effects."
        )
    requested = tuple(buffered.bounds)
    grid, window, truncated = build_grid(requested)
    if truncated:
        warnings.add(
            f"The requested context buffer is truncated by the DEM edge on the {', '.join(truncated)} side(s); "
            "flow routing near those sides lacks upstream context."
        )
    binfo["note"] = (
        "A buffer cannot guarantee correct upstream hydrology if the DEM does not extend sufficiently "
        "beyond the area of interest."
    )
    return SpatialDomain(
        pilot_full=pilot_full, pilot=inter, source_crs=src_crs, n_features=n_features, coverage_fraction=coverage,
        window=window, window_transform=grid.transform, extent_bounds=grid.bounds, requested_bounds=requested,
        truncated_sides=truncated, buffer_info=binfo, grid=grid, processing_crs=proc, resolution_info=res_info,
    )
