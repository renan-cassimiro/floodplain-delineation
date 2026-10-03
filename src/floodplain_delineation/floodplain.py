"""Potential floodplain raster generation from HAND."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask


def create_floodplain(
    hand_path: Path,
    pilot_area,
    output_path: Path,
    threshold_m: float,
) -> None:
    """Create a binary potential floodplain raster clipped to the pilot area."""
    with rasterio.open(hand_path) as src:
        geometries = [geometry for geometry in pilot_area.to_crs(src.crs).geometry]
        hand, transform = mask(src, geometries, crop=True, filled=False)
        hand_data = hand[0].astype(np.float32)
        valid = ~hand.mask[0]
        result = np.full(hand_data.shape, 255, dtype=np.uint8)
        result[valid] = (hand_data[valid] <= threshold_m).astype(np.uint8)

        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            count=1,
            height=result.shape[0],
            width=result.shape[1],
            transform=transform,
            nodata=255,
            compress="deflate",
            predictor=2,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(result, 1)
            dst.update_tags(
                classification="binary potential floodplain",
                value_0="outside potential floodplain",
                value_1="potential floodplain",
                nodata_value="255",
                hand_threshold_m=str(threshold_m),
                method="DEM-derived drainage network and HAND threshold",
                scientific_status="topographically defined potential floodplain; not observed inundation extent",
            )
