"""Stage 2: DEM extraction over the processing extent and hydrological conditioning.

Both steps work on full-width row stripes (see :mod:`rasters`), never on whole-raster arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

from .exceptions import InputError, ProcessingError
from .inputs import DemInfo, SpatialDomain
from .logging_utils import WarningCollector, get_logger
from .rasters import ensure_same_grid, grid_of, row_stripes, shift_rows, valid_mask
from .wbt import WhiteboxRunner

log = get_logger("preprocessing")

#: configuration method -> (WhiteboxTools tool name, extra flags)
CONDITIONING_TOOLS = {
    "breach": ("BreachDepressions", []),
    "fill": ("FillDepressionsWangAndLiu", ["--fix_flats"]),
}

#: Replacement nodata when the source DEM uses NaN (WhiteboxTools needs a finite sentinel).
NAN_NODATA_SENTINEL = -9999.0

#: Elevation changes below this magnitude (DEM vertical units) are not counted as modifications.
CHANGE_TOLERANCE = 1e-9
#: Thresholds (DEM vertical units) for the "cells lowered by more than X" diagnostic.
DECREASE_THRESHOLDS = (1.0, 5.0, 10.0, 25.0, 50.0)
#: Log-spaced bins for the approximate distribution of elevation decreases (20 bins per decade).
_DECREASE_EDGES = np.logspace(-3, 4, 141)


def extract_processing_dem(
    dem: DemInfo, domain: SpatialDomain, out_path: Path, warnings: WarningCollector
) -> Dict[str, Any]:
    """Write the unconditioned DEM over the processing window (original grid, no resampling).

    Valid values are copied unchanged. Integer DEMs are stored as float32 (lossless for
    elevation ranges); a NaN nodata is replaced by an explicit finite sentinel.
    """
    win = domain.window
    height, width = int(win.height), int(win.width)
    with rasterio.open(dem.path) as src:
        src_dtype = np.dtype(src.dtypes[0])
        floating = np.issubdtype(src_dtype, np.floating)
        out_dtype = src_dtype if floating else np.dtype(np.float32)
        if not floating:
            log.info("Integer DEM stored as float32 in the intermediate product (values unchanged).")
        out_nodata = dem.nodata
        if not np.isfinite(out_nodata):
            out_nodata = NAN_NODATA_SENTINEL
            warnings.add(f"DEM nodata is NaN/inf; intermediate rasters use {NAN_NODATA_SENTINEL} as nodata.")
        src_nodata = dem.nodata if np.isfinite(dem.nodata) else None

        profile: Dict[str, Any] = dict(
            driver="GTiff", height=height, width=width, count=1, dtype=out_dtype.name,
            crs=dem.crs, transform=domain.window_transform, nodata=out_nodata,
            BIGTIFF="IF_SAFER",  # uncompressed: read directly by WhiteboxTools
        )
        n_valid = n_nodata = n_inside = n_pilot_valid = 0
        pilot_miny, pilot_maxy = domain.pilot.bounds[1], domain.pilot.bounds[3]
        try:
            with rasterio.open(out_path, "w", **profile) as dst:
                for stripe in row_stripes(height, width, src_dtype.itemsize + out_dtype.itemsize + 2):
                    r0, rows = int(stripe.row_off), int(stripe.height)
                    read_win = Window(win.col_off, win.row_off + r0, width, rows)
                    arr = src.read(1, window=read_win)
                    valid = valid_mask(arr, src_nodata)
                    arr = np.where(valid, arr.astype(out_dtype, copy=False), out_dtype.type(out_nodata))
                    dst.write(arr, 1, window=stripe)
                    n_valid += int(valid.sum())
                    n_nodata += int(valid.size - valid.sum())

                    t = shift_rows(domain.window_transform, r0)
                    top, bottom = t.f, t.f + t.e * rows
                    if bottom <= pilot_maxy and top >= pilot_miny:  # stripe overlaps the pilot's y-range
                        inside = ~geometry_mask([domain.pilot], out_shape=arr.shape, transform=t, all_touched=False)
                        n_inside += int(inside.sum())
                        n_pilot_valid += int((inside & valid).sum())
                dst.update_tags(
                    PRODUCT="unconditioned DEM over processing extent", SOURCE=str(dem.path)
                )
            if n_valid == 0:
                raise InputError("The processing extent contains no valid (non-nodata) DEM cells.")
            # Pilot-area coverage of valid cells (fail early when the pilot only covers nodata).
            if n_inside == 0:
                raise InputError("The pilot area is smaller than one DEM cell; no cell centre falls inside it.")
            if n_pilot_valid == 0:
                raise InputError("All DEM cells inside the pilot area are nodata.")
        except Exception:
            out_path.unlink(missing_ok=True)
            raise
    if n_pilot_valid < n_inside:
        warnings.add(
            f"{n_inside - n_pilot_valid} of {n_inside} cells inside the pilot area "
            "are DEM nodata; they stay nodata in the final map."
        )
    return {
        "path": str(out_path),
        "shape": [height, width],
        "valid_cells": n_valid,
        "nodata_cells": n_nodata,
        "nodata_value": out_nodata,
        "dtype": out_dtype.name,
        "pilot_cells": n_inside,
        "pilot_valid_cells": n_pilot_valid,
    }


def _approx_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Upper bin edge at which the cumulative count first reaches ``q`` (conservative)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    idx = int(np.searchsorted(np.cumsum(counts), q * total, side="left"))
    return float(edges[min(idx + 1, len(edges) - 1)])


def compare_conditioned(
    dem_path: Path, cond_path: Path, tool: str, extra: Any
) -> Dict[str, Any]:
    """Compare the unconditioned and conditioned DEM block by block.

    Raises when the conditioned raster is not floating point, is all nodata, or changed the
    nodata footprint. Otherwise returns modification statistics, including how deep the
    conditioning cut (an unbounded breach can lower the terrain by tens of metres).
    """
    n_cond_valid = n_orig_valid = n_changed = n_to_data = n_to_nodata = 0
    max_dec = max_inc = 0.0
    n_lowered = 0
    counts = np.zeros(len(_DECREASE_EDGES) - 1, dtype=np.int64)
    exceed = {t: 0 for t in DECREASE_THRESHOLDS}
    with rasterio.open(dem_path) as a, rasterio.open(cond_path) as b:
        if not np.issubdtype(np.dtype(b.dtypes[0]), np.floating):
            raise ProcessingError(f"Conditioned DEM has unexpected data type {b.dtypes[0]}.")
        cond_dtype = b.dtypes[0]
        for win in row_stripes(a.height, a.width, 40):
            orig, cond = a.read(1, window=win), b.read(1, window=win)
            ov, cv = valid_mask(orig, a.nodata), valid_mask(cond, b.nodata)
            n_cond_valid += int(cv.sum())
            n_orig_valid += int(ov.sum())
            mismatch = cv != ov
            if mismatch.any():
                n_to_data += int((mismatch & ~ov).sum())
                n_to_nodata += int((mismatch & ov).sum())
                continue
            delta = cond[ov].astype(np.float64) - orig[ov].astype(np.float64)
            if not delta.size:
                continue
            n_changed += int((np.abs(delta) > CHANGE_TOLERANCE).sum())
            lowered = -delta[delta < -CHANGE_TOLERANCE]
            if lowered.size:
                n_lowered += int(lowered.size)
                max_dec = max(max_dec, float(lowered.max()))
                counts += np.histogram(np.clip(lowered, _DECREASE_EDGES[0], _DECREASE_EDGES[-1]), bins=_DECREASE_EDGES)[0]
                for t in DECREASE_THRESHOLDS:
                    exceed[t] += int((lowered > t).sum())
            if delta.max() > CHANGE_TOLERANCE:
                max_inc = max(max_inc, float(delta.max()))
    if n_cond_valid == 0:
        raise ProcessingError("Conditioning produced an all-nodata raster.")
    if n_to_data or n_to_nodata:
        raise ProcessingError(
            f"Conditioning ('{tool}') changed the nodata footprint of the DEM in {n_to_data + n_to_nodata} cell(s) "
            f"({n_to_data} nodata cells became data, {n_to_nodata} data cells became nodata). Nodata semantics "
            "must be preserved. Known cause: depression-filling tools may fill or corrupt interior nodata holes. "
            "Use conditioning.method 'breach', or fix/fill nodata holes in the DEM beforehand."
        )
    return {
        "tool": tool,
        "flags": extra,
        "valid_cells": n_cond_valid,
        "cells_modified": n_changed,
        "fraction_modified": n_changed / n_orig_valid if n_orig_valid else 0.0,
        "max_elevation_decrease": max_dec,
        "max_elevation_increase": max_inc,
        "cells_lowered": n_lowered,
        "decrease_percentiles_approx": {
            "p50": _approx_quantile(counts, _DECREASE_EDGES, 0.50),
            "p95": _approx_quantile(counts, _DECREASE_EDGES, 0.95),
            "p99": _approx_quantile(counts, _DECREASE_EDGES, 0.99),
            "note": "among lowered cells; log-binned (<=12% relative resolution), DEM vertical units",
        },
        "cells_lowered_more_than": {f"{t:g}": exceed[t] for t in DECREASE_THRESHOLDS},
        "output_dtype": cond_dtype,
    }


def condition_dem(
    runner: WhiteboxRunner,
    method: str,
    dem_path: Path,
    out_path: Path,
    work_dir: Path,
    warnings: WarningCollector,
) -> Dict[str, Any]:
    """Run hydrological conditioning and validate the result (block by block)."""
    tool, extra = CONDITIONING_TOOLS[method]
    log.info("Conditioning DEM with WhiteboxTools %s (method='%s').", tool, method)
    runner.run_raster(tool, [f"--dem={dem_path}", *extra], work_dir=work_dir, output=out_path)
    ensure_same_grid(grid_of(dem_path), out_path, "Conditioned DEM", warnings)
    diag = compare_conditioned(dem_path, out_path, tool, extra)
    log.info(
        "Conditioning modified %d cells (%.3f%%); max decrease %.3f, max increase %.3f (DEM vertical units).",
        diag["cells_modified"], 100 * diag["fraction_modified"],
        diag["max_elevation_decrease"], diag["max_elevation_increase"],
    )
    pct = diag["decrease_percentiles_approx"]
    log.info(
        "Cells lowered: %d; decrease p50 ~%.3f, p95 ~%.3f, p99 ~%.3f; lowered by more than %s.",
        diag["cells_lowered"], pct["p50"], pct["p95"], pct["p99"],
        ", ".join(f"{k}: {v}" for k, v in diag["cells_lowered_more_than"].items()),
    )
    return diag
