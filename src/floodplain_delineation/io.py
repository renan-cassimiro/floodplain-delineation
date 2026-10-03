"""Input/output helpers for rasters and vector data."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from .crs import validate_input_crs
from .exceptions import InputError


def read_pilot_area(path: Path, layer: str, configured_crs: CRS) -> gpd.GeoDataFrame:
    """Read, validate, and dissolve the pilot area geometries."""
    if not path.exists():
        raise InputError(f"Pilot-area GeoPackage does not exist: {path}")
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception as exc:
        raise InputError(f"Could not read pilot-area layer '{layer}': {exc}") from exc
    if gdf.empty:
        raise InputError("Pilot-area layer contains no features.")
    if gdf.crs is None:
        gdf = gdf.set_crs(configured_crs)
    else:
        validate_input_crs(gdf.crs, configured_crs, "Pilot area")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise InputError("Pilot-area layer contains no valid geometries.")
    if not gdf.geometry.is_valid.all():
        gdf.geometry = gdf.geometry.make_valid()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise InputError("Pilot-area layer contains no valid geometries after geometry repair.")
    return gpd.GeoDataFrame(
        {"geometry": [gdf.geometry.union_all()]},
        geometry="geometry",
        crs=configured_crs,
    )


def validate_dem(path: Path, configured_crs: CRS) -> None:
    """Validate basic DEM metadata."""
    if not path.exists():
        raise InputError(f"DEM file does not exist: {path}")
    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                raise InputError("DEM contains no raster bands.")
            if src.crs is not None:
                validate_input_crs(src.crs, configured_crs, "DEM")
            elif configured_crs is None:
                raise InputError("DEM has no CRS and no configured input CRS is available.")
            if src.width < 2 or src.height < 2:
                raise InputError("DEM is too small for hydrological processing.")
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Could not read DEM: {exc}") from exc


def write_geodataframe(gdf: gpd.GeoDataFrame, path: Path, layer: str = "drainage") -> None:
    """Write a GeoDataFrame to a GeoPackage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, layer=layer, driver="GPKG")
