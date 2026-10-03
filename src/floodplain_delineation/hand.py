"""HAND calculation using the DEM-derived drainage network."""

from __future__ import annotations

from pathlib import Path

import rasterio

from .exceptions import ProcessingError
from .hydrology import WhiteboxRunner


def calculate_hand(dem_path: Path, drainage_raster: Path, hand_path: Path) -> None:
    """Calculate elevation above the nearest downslope drainage cell."""
    runner = WhiteboxRunner(dem_path.parent)
    runner.run(
        "elevation_above_stream",
        [
            f"--dem={dem_path}",
            f"--streams={drainage_raster}",
            f"--output={hand_path}",
        ],
    )
    if not hand_path.exists():
        raise ProcessingError(f"HAND raster was not created: {hand_path}")
    with rasterio.open(hand_path, "r+") as dataset:
        dataset.update_tags(
            method="WhiteboxTools elevation_above_stream on DEM-derived drainage",
            units="m",
            interpretation="elevation above nearest downslope drainage cell",
        )
