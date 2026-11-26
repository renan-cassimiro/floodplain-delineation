"""Stage 2: hydrological conditioning of the processing DEM.

The comparison with the unconditioned DEM works on full-width row stripes (see :mod:`rasters`),
never on whole-raster arrays. (Stage 0, :mod:`reprojection`, produces the processing DEM.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio

from .config import ProcessingConfig
from .exceptions import ProcessingError
from .logging_utils import WarningCollector, get_logger
from .rasters import ensure_same_grid, grid_of, row_stripes, valid_mask
from .wbt import WhiteboxRunner

log = get_logger("preprocessing")

#: methods with fixed flags: configuration method -> (WhiteboxTools tool name, extra flags)
CONDITIONING_TOOLS = {
    "breach": ("BreachDepressions", []),
    "fill": ("FillDepressionsWangAndLiu", ["--fix_flats"]),
}
BREACH_LEAST_COST_TOOL = "BreachDepressionsLeastCost"


FILL_AFTER_BREACH_TOOL = "FillDepressionsWangAndLiu"
FILL_AFTER_BREACH_FLAGS = ["--fix_flats"]


def conditioning_plan(processing: ProcessingConfig) -> List[Tuple[str, List[str]]]:
    """Ordered WhiteboxTools steps ``[(tool, flags), ...]`` for the configured conditioning method.

    ``breach_least_cost`` maps the configuration one-to-one onto the flags of ``BreachDepressionsLeastCost``
    (checked against the installed tool's help by the runner): ``--dist`` (maximum breach path length, cells),
    ``--max_cost``, ``--min_dist`` (flag) and ``--flat_increment``. Nothing is passed that was not configured.

    ``fill_remaining`` adds a second step, ``FillDepressionsWangAndLiu --fix_flats`` on the breached DEM (breach
    first, then fill what is left). The tool's own ``--fill`` flag is deliberately not used: in WhiteboxTools 2.0.0
    it panics ("Error unwrapping 'output'") whenever a depression is left to fill (reproduced), whereas two explicit
    steps behave the same on every version and let each step be validated and reported separately.
    """
    method = processing.conditioning_method
    if method != "breach_least_cost":
        tool, flags = CONDITIONING_TOOLS[method]
        return [(tool, list(flags))]
    bl = processing.breach_least_cost
    if bl is None:  # pragma: no cover - rejected by the configuration parser
        raise ProcessingError("breach_least_cost conditioning needs 'processing.conditioning.breach_least_cost'.")
    flags = [f"--dist={bl.max_distance_cells}"]
    if bl.max_cost is not None:
        flags.append(f"--max_cost={bl.max_cost!r}")
    if bl.minimize_distance:
        flags.append("--min_dist")
    if bl.flat_increment is not None:
        flags.append(f"--flat_increment={bl.flat_increment!r}")
    steps = [(BREACH_LEAST_COST_TOOL, flags)]
    if processing.fill_remaining:
        steps.append((FILL_AFTER_BREACH_TOOL, list(FILL_AFTER_BREACH_FLAGS)))
    return steps

#: Elevation changes below this magnitude (DEM vertical units) are not counted as modifications.
CHANGE_TOLERANCE = 1e-9
#: Thresholds (DEM vertical units) for the "cells lowered by more than X" diagnostic.
DECREASE_THRESHOLDS = (1.0, 5.0, 10.0, 25.0, 50.0)
#: Log-spaced bins for the approximate distribution of elevation decreases (20 bins per decade).
_DECREASE_EDGES = np.logspace(-3, 4, 141)


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


def _log_diag(diag: Dict[str, Any]) -> None:
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


def condition_dem(
    runner: WhiteboxRunner,
    method: str,
    steps: List[Tuple[str, List[str]]],
    dem_path: Path,
    out_path: Path,
    work_dir: Path,
    warnings: WarningCollector,
) -> Dict[str, Any]:
    """Run the conditioning steps in order and validate the result block by block.

    A single step writes ``out_path``. With a second step (breach, then fill) the first one writes an
    intermediate raster in ``work_dir``; its effect and the effect of the fill are reported separately and the
    intermediate raster is removed. The reported modification statistics always compare the final conditioned DEM
    with the original one.
    """
    grid = grid_of(dem_path)
    current = dem_path
    step_reports: List[Dict[str, Any]] = []
    temporaries: List[Path] = []
    try:
        for i, (tool, flags) in enumerate(steps):
            last = i == len(steps) - 1
            target = out_path if last else work_dir / f"conditioning_step{i + 1}.tif"
            log.info("Conditioning step %d/%d with WhiteboxTools %s (method='%s', flags: %s).",
                     i + 1, len(steps), tool, method, " ".join(flags) or "none")
            runner.run_raster(tool, [f"--dem={current}", *flags], work_dir=work_dir, output=target)
            if not last:
                temporaries.append(target)
            ensure_same_grid(grid, target, f"Conditioned DEM (step {i + 1})", warnings)
            if len(steps) > 1:
                step_diag = compare_conditioned(current, target, tool, flags)
                step_reports.append({
                    "step": i + 1, "tool": tool, "flags": flags,
                    "cells_modified": step_diag["cells_modified"],
                    "max_elevation_decrease": step_diag["max_elevation_decrease"],
                    "max_elevation_increase": step_diag["max_elevation_increase"],
                })
                log.info("  step %d (%s): %d cells modified, max decrease %.3f, max increase %.3f.", i + 1, tool,
                         step_diag["cells_modified"], step_diag["max_elevation_decrease"], step_diag["max_elevation_increase"])
            current = target
        diag = compare_conditioned(dem_path, out_path, steps[-1][0], steps[-1][1])
    finally:
        for t in temporaries:
            t.unlink(missing_ok=True)
    diag["tools"] = [t for t, _ in steps]
    diag["tool"] = " + ".join(diag["tools"])
    diag["steps"] = step_reports or [{"step": 1, "tool": steps[0][0], "flags": steps[0][1]}]
    _log_diag(diag)
    return diag
