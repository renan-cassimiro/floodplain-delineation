"""Stage 0: DEM onto the master processing grid (``dem_processing_crs.tif``).

Everything downstream (conditioning, D8, accumulation, drainage, HAND, basins, REM) consumes only this
raster. The source DEM file is never modified.

Two cases, decided in :func:`inputs.prepare_spatial_domain`:

* the DEM is already in the processing CRS at the processing resolution: the master grid is a window of
  the DEM grid and the values are copied unchanged (no resampling);
* otherwise the DEM is warped onto the master grid with ``processing.dem_resampling``. Reprojection and
  resampling change the elevation values and therefore can change every hydrological result.

Nodata policy: GDAL's resamplers renormalise over the valid source neighbours, which would extend the
valid area by up to about one source cell. To avoid creating data where the DEM has none (and to never
interpolate across nodata areas), an output cell is valid only if its interpolated value is valid AND the
nearest source cell is valid.

The warp is executed in full-width row stripes, so memory does not depend on the raster size.

Reproducibility of the warp. GDAL chooses the width of the bilinear/cubic kernel from the size of the destination
window it is given; when source and target resolutions are close (the usual case here, e.g. 30 m -> 29.9 m) values
therefore differ slightly between decompositions (measured on synthetic data: from ~1e-4 m with one window to a few
centimetres with a few hundred rows; the same behaviour as ``gdalwarp`` with different ``-wm``). To make the result
independent of the memory budget (``rasters.BLOCK_BYTES``), the warp stripe height is a fixed function of the grid
width (:func:`warp_stripe_rows`) and is part of the stage fingerprint. Window copies (same CRS and resolution) are
exact and use the memory-budget stripes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from . import resources
from .exceptions import InputError, ProcessingError
from .inputs import DemInfo, SpatialDomain
from .logging_utils import WarningCollector, get_logger
from .rasters import create_raster, row_stripes, shift_rows, valid_mask

log = get_logger("reprojection")

#: Replacement nodata when the source DEM uses NaN (WhiteboxTools needs a finite sentinel).
NAN_NODATA_SENTINEL = -9999.0

_RESAMPLING = {
    "near": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
}

#: Number of GDAL warp threads (the work is chunked by stripes, one warp call at a time).
WARP_THREADS = max(1, min(8, resources.cpu_count()))

#: Fixed warp stripe geometry (see the module docstring). Changing either value changes the warped values, so
#: both are part of the DEM-reprojection fingerprint.
WARP_STRIPE_ROWS = 1024
WARP_MAX_STRIPE_CELLS = 2 ** 27       # ~134 M cells: bounds the working memory of one warp call for very wide grids
WARP_MEM_LIMIT_MB = 1024              # GDAL warp memory limit per call (avoids GDAL's own sub-chunking of a stripe)


def warp_stripe_rows(width: int) -> int:
    """Rows per warp stripe: a deterministic function of the grid width, never of the memory budget."""
    return int(max(64, min(WARP_STRIPE_ROWS, WARP_MAX_STRIPE_CELLS // max(1, width))))


def warp_settings() -> Dict[str, Any]:
    """Settings that determine the warped values (for the stage fingerprint and the summary)."""
    return {"stripe_rows": WARP_STRIPE_ROWS, "max_stripe_cells": WARP_MAX_STRIPE_CELLS, "mem_limit_mb": WARP_MEM_LIMIT_MB}


def _source_window_for(grid_bounds, dem: DemInfo, grid_crs) -> Window:
    """Window of the source DEM covering the master-grid extent (edges densified), clipped to the DEM."""
    left, bottom, right, top = transform_bounds(grid_crs, dem.crs, *grid_bounds, densify_pts=21)
    w = from_bounds(left, bottom, right, top, transform=dem.transform)
    c0, r0 = max(0, int(np.floor(w.col_off))), max(0, int(np.floor(w.row_off)))
    c1, r1 = min(dem.width, int(np.ceil(w.col_off + w.width))), min(dem.height, int(np.ceil(w.row_off + w.height)))
    return Window(c0, r0, max(0, c1 - c0), max(0, r1 - r0))


def _source_statistics(dem: DemInfo, window: Window) -> Dict[str, Any]:
    """Valid-cell count and elevation min/max/mean of the source DEM over ``window`` (block by block)."""
    n, lo, hi, total = 0, np.inf, -np.inf, 0.0
    nodata = dem.nodata if np.isfinite(dem.nodata) else None
    if window.width <= 0 or window.height <= 0:
        return {"valid_cells": 0}
    with rasterio.open(dem.path) as src:
        for stripe in row_stripes(int(window.height), int(window.width), np.dtype(src.dtypes[0]).itemsize + 9):
            win = Window(window.col_off, window.row_off + stripe.row_off, window.width, stripe.height)
            arr = src.read(1, window=win)
            v = arr[valid_mask(arr, nodata)].astype(np.float64)
            if v.size:
                n += int(v.size)
                lo, hi, total = min(lo, float(v.min())), max(hi, float(v.max())), total + float(v.sum())
    return {"valid_cells": n, "min": None if n == 0 else lo, "max": None if n == 0 else hi,
            "mean": None if n == 0 else total / n}


def reproject_dem(
    dem: DemInfo, domain: SpatialDomain, out_path: Path, resampling: str, warnings: WarningCollector
) -> Dict[str, Any]:
    """Write the DEM on the master grid and validate it. Returns the stage diagnostics."""
    grid = domain.grid
    if grid is None:
        raise ProcessingError("Internal error: the spatial domain has no processing grid.")
    method = _RESAMPLING[resampling]
    width, height = grid.width, grid.height
    with rasterio.open(dem.path) as src:
        src_dtype = np.dtype(src.dtypes[0])
        floating = np.issubdtype(src_dtype, np.floating)
        out_dtype = src_dtype if floating else np.dtype(np.float32)
        if not floating:
            log.info("Integer DEM stored as float32 in the processing DEM (values unchanged where not resampled).")
        out_nodata = dem.nodata
        if not np.isfinite(out_nodata):
            out_nodata = NAN_NODATA_SENTINEL
            warnings.add(f"DEM nodata is NaN/inf; the processing DEM uses {NAN_NODATA_SENTINEL} as nodata.")
        src_nodata = dem.nodata if np.isfinite(dem.nodata) else float("nan")

        copy_mode = grid.identical_to_source
        if copy_mode:
            log.info("Processing grid equals a window of the DEM grid: values are copied, no resampling.")
        else:
            log.info(
                "Reprojecting the DEM to %s at %.6g x %.6g (%s resampling, nearest-neighbour validity footprint).",
                domain.processing_crs.name, grid.res_x, grid.res_y, resampling,
            )
            warnings.add(
                f"The DEM was reprojected/resampled ('{resampling}') to the processing grid; elevation values "
                "change and so may all hydrological results compared with runs in the original CRS."
            )
        n_valid = n_inside = n_pilot_valid = 0
        v_lo, v_hi, v_sum = np.inf, -np.inf, 0.0
        pilot_miny, pilot_maxy = domain.pilot.bounds[1], domain.pilot.bounds[3]
        try:
            with create_raster(
                out_path, height=height, width=width, dtype=out_dtype.name, transform=grid.transform,
                crs=grid.crs, nodata=out_nodata,
                tags={"PRODUCT": "DEM on the master processing grid", "SOURCE": str(dem.path),
                      "RESAMPLING": "none (window copy)" if copy_mode else resampling},
            ) as dst:
                if copy_mode:
                    stripes = row_stripes(height, width, src_dtype.itemsize + out_dtype.itemsize * 3 + 4)
                else:
                    step = warp_stripe_rows(width)
                    stripes = (Window(0, r, width, min(step, height - r)) for r in range(0, height, step))
                for stripe in stripes:
                    r0, rows = int(stripe.row_off), int(stripe.height)
                    t = shift_rows(grid.transform, r0)
                    if copy_mode:
                        w0 = grid.source_window
                        arr = src.read(1, window=Window(w0.col_off, w0.row_off + r0, width, rows))
                        valid = valid_mask(arr, None if np.isnan(src_nodata) else src_nodata)
                        out = np.where(valid, arr.astype(out_dtype, copy=False), out_dtype.type(out_nodata))
                    else:
                        out = np.full((rows, width), out_nodata, dtype=out_dtype)
                        reproject(
                            source=rasterio.band(src, 1), destination=out, dst_transform=t, dst_crs=grid.crs,
                            src_nodata=src_nodata, dst_nodata=out_nodata, resampling=method,
                            init_dest_nodata=True, num_threads=WARP_THREADS,
                            warp_mem_limit=WARP_MEM_LIMIT_MB,
                        )
                        valid = valid_mask(out, out_nodata)
                        if method != Resampling.nearest:
                            near = np.full((rows, width), out_nodata, dtype=out_dtype)
                            reproject(
                                source=rasterio.band(src, 1), destination=near, dst_transform=t, dst_crs=grid.crs,
                                src_nodata=src_nodata, dst_nodata=out_nodata, resampling=Resampling.nearest,
                                init_dest_nodata=True, num_threads=WARP_THREADS,
                            warp_mem_limit=WARP_MEM_LIMIT_MB,
                            )
                            valid &= valid_mask(near, out_nodata)
                        out = np.where(valid, out, out_dtype.type(out_nodata))
                    dst.write(out, 1, window=stripe)
                    v = out[valid].astype(np.float64)
                    if v.size:
                        n_valid += int(v.size)
                        v_lo, v_hi, v_sum = min(v_lo, float(v.min())), max(v_hi, float(v.max())), v_sum + float(v.sum())
                    if t.f + t.e * rows <= pilot_maxy and t.f >= pilot_miny:
                        inside = ~geometry_mask([domain.pilot], out_shape=out.shape, transform=t, all_touched=False)
                        n_inside += int(inside.sum())
                        n_pilot_valid += int((inside & valid).sum())
                if n_valid == 0:
                    raise InputError("The processing grid contains no valid (non-nodata) DEM cells.")
                if n_inside == 0:
                    raise InputError("The pilot area is smaller than one processing cell; no cell centre falls inside it.")
                if n_pilot_valid == 0:
                    raise InputError("All DEM cells inside the pilot area are nodata.")
        except ProcessingError:
            raise
        except (OSError, rasterio.errors.RasterioError) as exc:
            raise ProcessingError(f"Reprojection of the DEM failed: {exc}") from exc
    if n_pilot_valid < n_inside:
        warnings.add(
            f"{n_inside - n_pilot_valid} of {n_inside} cells inside the pilot area "
            "are DEM nodata; they stay nodata in the final map."
        )

    after = {"valid_cells": n_valid, "min": float(v_lo), "max": float(v_hi), "mean": v_sum / n_valid}
    src_window = grid.source_window if copy_mode else _source_window_for(grid.bounds, dem, grid.crs)
    before = _source_statistics(dem, src_window)
    if before.get("valid_cells"):
        log.info(
            "DEM statistics before -> after: valid cells %d -> %d; min %.3f -> %.3f; max %.3f -> %.3f; mean %.3f -> %.3f.",
            before["valid_cells"], after["valid_cells"], before["min"], after["min"],
            before["max"], after["max"], before["mean"], after["mean"],
        )
    return {
        "path": str(out_path),
        "shape": [height, width],
        "grid": grid.describe(),
        "resampling": "none (window copy: same CRS and resolution)" if copy_mode else resampling,
        "validity_rule": None if copy_mode else
        "output cell valid only if the interpolated value is valid and the nearest source cell is valid",
        "valid_cells": n_valid,
        "nodata_cells": width * height - n_valid,
        "nodata_value": out_nodata,
        "dtype": out_dtype.name,
        "statistics_after": after,
        "statistics_before_over_source_window": before,
        "pilot_cells": n_inside,
        "pilot_valid_cells": n_pilot_valid,
        "gdal_version": rasterio.__gdal_version__,
        "warp_settings": None if copy_mode else {**warp_settings(), "stripe_rows_used": warp_stripe_rows(width)},
        "elevation_changed_by_resampling": not copy_mode,
    }
