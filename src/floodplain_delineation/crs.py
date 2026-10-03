"""CRS validation and raster reprojection utilities."""

from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pyproj import CRS

from .exceptions import ConfigurationError, InputError


def validate_input_crs(source_crs: CRS | None, configured_crs: CRS, dataset_name: str) -> CRS:
    """Check a source CRS against the configured input CRS."""
    if source_crs is None:
        return configured_crs
    if CRS.from_user_input(source_crs) != configured_crs:
        raise InputError(
            f"{dataset_name} CRS ({source_crs}) does not match configured input CRS "
            f"({configured_crs}). Configure the correct input CRS rather than relying on an implicit conversion."
        )
    return source_crs


def validate_processing_crs(crs: CRS) -> None:
    """Require a projected CRS with metre-based linear units."""
    if not crs.is_projected:
        raise ConfigurationError("Output CRS must be projected and use metric units.")
    units = [axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name]
    if not any("metre" in unit or "meter" in unit for unit in units):
        raise ConfigurationError("Output CRS must use metres as its linear unit.")


def reproject_raster(
    source_path: Path,
    destination_path: Path,
    source_crs: CRS,
    destination_crs: CRS,
) -> None:
    """Reproject a DEM while preserving its approximate source resolution."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise InputError(f"DEM has no CRS metadata: {source_path}")
        transform, width, height = calculate_default_transform(
            src.crs,
            destination_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            crs=destination_crs,
            transform=transform,
            width=width,
            height=height,
            compress="deflate",
            predictor=2 if src.dtypes[0].startswith("int") else 3,
        )
        with rasterio.open(destination_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=source_crs,
                dst_transform=transform,
                dst_crs=destination_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )
