"""Small raster I/O helpers with explicit grid, CRS and nodata checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine

from .exceptions import ProcessingError
from .logging_utils import get_logger, WarningCollector

log = get_logger("rasters")


@dataclass(frozen=True)
class GridInfo:
    height: int
    width: int
    transform: Affine
    crs: Optional[RioCRS]

    def same_grid(self, other: "GridInfo") -> bool:
        if (self.height, self.width) != (other.height, other.width):
            return False
        tol = 1e-6 * min(abs(self.transform.a), abs(self.transform.e))
        return all(abs(x - y) <= tol for x, y in zip(tuple(self.transform)[:6], tuple(other.transform)[:6]))


def valid_mask(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """True where a cell holds valid data (finite and not equal to nodata)."""
    mask = np.ones(array.shape, dtype=bool)
    if np.issubdtype(array.dtype, np.floating):
        mask &= np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= array != array.dtype.type(nodata)
    return mask


def read_band(path: Path) -> Tuple[np.ndarray, np.ndarray, GridInfo, Optional[float]]:
    """Read band 1 -> (array, valid_mask, grid, nodata)."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            grid = GridInfo(src.height, src.width, src.transform, src.crs)
            nodata = src.nodata
    except Exception as exc:
        raise ProcessingError(f"Cannot read raster {path}: {exc}") from exc
    return arr, valid_mask(arr, nodata), grid, nodata


def grid_of(path: Path) -> GridInfo:
    with rasterio.open(path) as src:
        return GridInfo(src.height, src.width, src.transform, src.crs)


def ensure_same_grid(
    reference: GridInfo, path: Path, what: str, warnings: Optional[WarningCollector] = None
) -> None:
    """Verify that ``path`` is aligned with ``reference`` (no silent resampling).

    A missing CRS (some tools drop it) is re-stamped from the reference when the
    pixel grid is identical; this is logged and recorded as a warning. Any grid
    mismatch is an error.
    """
    try:
        found = grid_of(path)
    except Exception as exc:
        raise ProcessingError(f"{what}: output raster {path} is unreadable: {exc}") from exc
    if not reference.same_grid(found):
        raise ProcessingError(
            f"{what}: output grid differs from the DEM grid (shape {found.height}x{found.width} vs "
            f"{reference.height}x{reference.width} or transform mismatch). The workflow never "
            "resamples; this indicates a failed or incompatible tool run."
        )
    if found.crs is None and reference.crs is not None:
        with rasterio.open(path, "r+") as dst:
            dst.crs = reference.crs
        msg = f"{what}: tool output had no CRS; re-assigned the DEM CRS (grid identical)."
        (warnings.add(msg) if warnings else log.warning(msg))
    elif reference.crs is not None and found.crs != reference.crs:
        raise ProcessingError(f"{what}: output CRS differs from the DEM CRS ({found.crs} vs {reference.crs}).")


def write_raster(
    path: Path,
    array: np.ndarray,
    *,
    transform: Affine,
    crs: RioCRS,
    nodata: Optional[float],
    compress: Optional[str] = "deflate",
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> None:
    """Write a single-band GeoTIFF."""
    profile: Dict[str, Any] = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        BIGTIFF="IF_SAFER",
    )
    if compress:
        profile["compress"] = compress
    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)
            if tags:
                dst.update_tags(**{k: str(v) for k, v in tags.items()})
            if description:
                dst.set_band_description(1, description)
    except Exception as exc:
        raise ProcessingError(f"Cannot write raster {path}: {exc}") from exc
