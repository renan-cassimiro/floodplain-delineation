"""Small raster I/O helpers with explicit grid, CRS and nodata checks."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine
from rasterio.windows import Window

from .exceptions import FloodplainError, ProcessingError
from .logging_utils import get_logger, WarningCollector

log = get_logger("rasters")

#: Memory budget (bytes) for the arrays held at once while processing a raster block by block.
#: Whole-raster arrays are never materialised in the processing stages: a 900 M-cell DEM is
#: handled as full-width row stripes, so Python memory stays independent of the raster size.
BLOCK_BYTES = 192 * 2 ** 20


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


def block_rows(width: int, bytes_per_cell: int, align: int = 1) -> int:
    """Number of rows per full-width stripe so that ``rows*width*bytes_per_cell <= BLOCK_BYTES``.

    ``align`` rounds the result down to a multiple (e.g. the tile height of an output), min 1 block.
    """
    rows = max(1, int(BLOCK_BYTES // max(1, width * bytes_per_cell)))
    if align > 1:
        rows = max(align, (rows // align) * align)
    return rows


def row_stripes(height: int, width: int, bytes_per_cell: int, align: int = 1) -> Iterator[Window]:
    """Full-width row stripes covering ``height`` rows."""
    step = block_rows(width, bytes_per_cell, align)
    for r0 in range(0, height, step):
        yield Window(0, r0, width, min(step, height - r0))


def shift_rows(transform: Affine, rows: int) -> Affine:
    """Transform of a window that starts ``rows`` rows below the origin of ``transform``."""
    return Affine(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f + transform.e * rows)


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


def normalize_geotiff(src_path: Path, dst_path: Path) -> None:
    """Rewrite a GeoTIFF with GDAL, block by block, preserving grid, dtype, nodata, CRS and tags.

    The output is a plain uncompressed GeoTIFF (BigTIFF when needed) with the same layout as
    the DEM written by :func:`write_raster`, a profile WhiteboxTools reads reliably. The data
    are copied unchanged. Written to a temporary name and moved into place only when complete.
    """
    tmp = dst_path.with_name(dst_path.name + ".partial")
    try:
        with rasterio.open(src_path) as src:
            profile: Dict[str, Any] = dict(
                driver="GTiff", height=src.height, width=src.width, count=1, dtype=src.dtypes[0],
                crs=src.crs, transform=src.transform, nodata=src.nodata, BIGTIFF="IF_SAFER",
            )
            itemsize = np.dtype(src.dtypes[0]).itemsize
            with rasterio.open(tmp, "w", **profile) as dst:
                for win in row_stripes(src.height, src.width, 2 * itemsize):
                    dst.write(src.read(1, window=win), 1, window=win)
                tags = src.tags()
                if tags:
                    dst.update_tags(**tags)
        os.replace(tmp, dst_path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise ProcessingError(f"Cannot rewrite raster {src_path} with GDAL: {exc}") from exc


def scan_min_max(path: Path, nodata_aware: bool = True) -> Tuple[Optional[float], Optional[float], int]:
    """(min, max, valid_cell_count) of band 1 over valid cells, read block by block."""
    lo, hi, n = None, None, 0
    with rasterio.open(path) as src:
        for win in row_stripes(src.height, src.width, np.dtype(src.dtypes[0]).itemsize + 1):
            arr = src.read(1, window=win)
            v = arr[valid_mask(arr, src.nodata)] if nodata_aware else arr.ravel()
            if v.size:
                lo = float(v.min()) if lo is None else min(lo, float(v.min()))
                hi = float(v.max()) if hi is None else max(hi, float(v.max()))
                n += int(v.size)
    return lo, hi, n


@contextmanager
def create_raster(
    path: Path,
    *,
    height: int,
    width: int,
    dtype: str,
    transform: Affine,
    crs: Optional[RioCRS],
    nodata: Optional[float],
    compress: Optional[str] = None,
    tiled: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> Iterator[Any]:
    """Open a single-band GeoTIFF for block-wise writing; the partial file is removed on failure.

    ``tiled`` uses 512x512 tiles (callers should write stripes whose height is a multiple of 512).
    """
    profile: Dict[str, Any] = dict(
        driver="GTiff", height=height, width=width, count=1, dtype=dtype,
        crs=crs, transform=transform, nodata=nodata, BIGTIFF="IF_SAFER",
    )
    if compress:
        profile["compress"] = compress
    if tiled:
        profile.update(tiled=True, blockxsize=512, blockysize=512)
    try:
        with rasterio.open(path, "w", **profile) as dst:
            if tags:
                dst.update_tags(**{k: str(v) for k, v in tags.items()})
            if description:
                dst.set_band_description(1, description)
            yield dst
    except FloodplainError:
        Path(path).unlink(missing_ok=True)
        raise
    except (OSError, rasterio.errors.RasterioError) as exc:
        Path(path).unlink(missing_ok=True)
        raise ProcessingError(f"Cannot write raster {path}: {exc}") from exc
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
