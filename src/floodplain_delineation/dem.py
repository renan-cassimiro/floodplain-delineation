"""DEM preparation and pilot-area clipping."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import box

from .exceptions import InputError


def prepare_dem(
    source_dem: Path,
    pilot_area: gpd.GeoDataFrame,
    output_path: Path,
    input_crs,
    buffer_m: float,
) -> None:
    """Crop the DEM to a buffered pilot-area bounding region in input CRS."""
    with rasterio.open(source_dem) as src:
        src_crs = src.crs or input_crs
        pilot = pilot_area.to_crs(input_crs)
        buffered = pilot.geometry.iloc[0].buffer(buffer_m)
        bbox = buffered.bounds
        if not box(*bbox).intersects(box(*transform_bounds(src_crs, input_crs, *src.bounds))):
            raise InputError("DEM and pilot area do not overlap.")

        bbox_src = transform_bounds(input_crs, src_crs, *bbox, densify_pts=21)
        crop_geometry = box(*bbox_src)
        try:
            data, transform = mask(src, [crop_geometry], crop=True, filled=True)
        except ValueError as exc:
            raise InputError("DEM and pilot area do not overlap after CRS transformation.") from exc

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=src_crs,
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            compress="deflate",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
