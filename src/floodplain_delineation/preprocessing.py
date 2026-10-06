"""Stage 2: DEM extraction over the processing extent and hydrological conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.features import geometry_mask

from .exceptions import InputError, ProcessingError
from .inputs import DemInfo, SpatialDomain
from .logging_utils import WarningCollector, get_logger
from .rasters import GridInfo, ensure_same_grid, read_band, valid_mask, write_raster
from .wbt import WhiteboxRunner

log = get_logger("preprocessing")

#: configuration method -> (WhiteboxTools tool name, extra flags)
CONDITIONING_TOOLS = {
    "breach": ("BreachDepressions", []),
    "fill": ("FillDepressionsWangAndLiu", ["--fix_flats"]),
}

#: Replacement nodata when the source DEM uses NaN (WhiteboxTools needs a finite sentinel).
NAN_NODATA_SENTINEL = -9999.0


def extract_processing_dem(
    dem: DemInfo, domain: SpatialDomain, out_path: Path, warnings: WarningCollector
) -> Dict[str, Any]:
    """Write the unconditioned DEM over the processing window (original grid, no resampling).

    Valid values are copied unchanged. Integer DEMs are stored as float32 (lossless for
    elevation ranges); a NaN nodata is replaced by an explicit finite sentinel.
    """
    with rasterio.open(dem.path) as src:
        arr = src.read(1, window=domain.window)
    valid = valid_mask(arr, dem.nodata if np.isfinite(dem.nodata) else None)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise InputError("The processing extent contains no valid (non-nodata) DEM cells.")

    out_nodata = dem.nodata
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
        log.info("Integer DEM stored as float32 in the intermediate product (values unchanged).")
    if not np.isfinite(out_nodata):
        out_nodata = NAN_NODATA_SENTINEL
        warnings.add(f"DEM nodata is NaN/inf; intermediate rasters use {NAN_NODATA_SENTINEL} as nodata.")
    bad = ~valid
    arr = np.where(bad, arr.dtype.type(out_nodata), arr)

    # Pilot-area coverage of valid cells (fail early when the pilot only covers nodata).
    inside = ~geometry_mask([domain.pilot], out_shape=arr.shape, transform=domain.window_transform, all_touched=False)
    if not inside.any():
        raise InputError("The pilot area is smaller than one DEM cell; no cell centre falls inside it.")
    n_pilot_valid = int((inside & valid).sum())
    if n_pilot_valid == 0:
        raise InputError("All DEM cells inside the pilot area are nodata.")
    if n_pilot_valid < inside.sum():
        warnings.add(
            f"{int(inside.sum() - n_pilot_valid)} of {int(inside.sum())} cells inside the pilot area "
            "are DEM nodata; they stay nodata in the final map."
        )

    write_raster(
        out_path, arr, transform=domain.window_transform, crs=dem.crs, nodata=out_nodata,
        compress=None,  # uncompressed: read directly by WhiteboxTools
        tags={"PRODUCT": "unconditioned DEM over processing extent", "SOURCE": str(dem.path)},
    )
    return {
        "path": str(out_path),
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "valid_cells": n_valid,
        "nodata_cells": int(bad.sum()),
        "nodata_value": out_nodata,
        "dtype": str(arr.dtype),
        "pilot_cells": int(inside.sum()),
        "pilot_valid_cells": n_pilot_valid,
    }


def condition_dem(
    runner: WhiteboxRunner,
    method: str,
    dem_path: Path,
    out_path: Path,
    work_dir: Path,
    warnings: WarningCollector,
) -> Tuple[Dict[str, Any], np.ndarray, GridInfo]:
    """Run hydrological conditioning and validate the result.

    Returns ``(diagnostics, valid_mask, grid)`` of the conditioned DEM.
    """
    tool, extra = CONDITIONING_TOOLS[method]
    log.info("Conditioning DEM with WhiteboxTools %s (method='%s').", tool, method)
    runner.run(tool, [f"--dem={dem_path}", f"--output={out_path}", *extra], work_dir=work_dir, outputs=[out_path])

    orig, orig_valid, grid, orig_nodata = read_band(dem_path)
    ensure_same_grid(grid, out_path, "Conditioned DEM", warnings)
    cond, cond_valid, _, cond_nodata = read_band(out_path)

    if not np.issubdtype(cond.dtype, np.floating):
        raise ProcessingError(f"Conditioned DEM has unexpected data type {cond.dtype}.")
    if int(cond_valid.sum()) == 0:
        raise ProcessingError("Conditioning produced an all-nodata raster.")
    mismatch = cond_valid != orig_valid
    if mismatch.any():
        raise ProcessingError(
            f"Conditioning ('{tool}') changed the nodata footprint of the DEM in {int(mismatch.sum())} cell(s) "
            f"({int((mismatch & ~orig_valid).sum())} nodata cells became data, "
            f"{int((mismatch & orig_valid).sum())} data cells became nodata). Nodata semantics must be "
            "preserved. Known cause: depression-filling tools may fill or corrupt interior nodata holes. "
            "Use conditioning.method 'breach', or fix/fill nodata holes in the DEM beforehand."
        )
    delta = cond[orig_valid].astype(np.float64) - orig[orig_valid].astype(np.float64)
    changed = np.abs(delta) > 1e-9
    diag = {
        "tool": tool,
        "flags": extra,
        "valid_cells": int(cond_valid.sum()),
        "cells_modified": int(changed.sum()),
        "fraction_modified": float(changed.mean()),
        "max_elevation_decrease": float(-delta.min()) if delta.size and delta.min() < 0 else 0.0,
        "max_elevation_increase": float(delta.max()) if delta.size and delta.max() > 0 else 0.0,
        "output_dtype": str(cond.dtype),
    }
    log.info(
        "Conditioning modified %d cells (%.3f%%); max decrease %.3f, max increase %.3f (DEM vertical units).",
        diag["cells_modified"], 100 * diag["fraction_modified"],
        diag["max_elevation_decrease"], diag["max_elevation_increase"],
    )
    return diag, cond_valid, grid
