"""Stage 5: Height Above Nearest Drainage (HAND).

HAND(x) = Z(x) - Z_drain(x), with Z_drain the elevation of the nearest drainage cell
reached by following the D8 flow path downslope (WhiteboxTools ``ElevationAboveStream``).
Units are the vertical units of the DEM. Drainage cells have HAND = 0.
Cells that never reach a drainage cell (e.g. drain off the extent) are nodata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .exceptions import ProcessingError
from .logging_utils import WarningCollector, get_logger
from .rasters import GridInfo, ensure_same_grid, read_band
from .wbt import WhiteboxRunner

log = get_logger("hand")


def compute_hand(
    runner: WhiteboxRunner, cond_dem: Path, drainage: Path, out_path: Path, work_dir: Path,
    dem_valid: np.ndarray, drain: np.ndarray, grid: GridInfo, warnings: WarningCollector,
) -> Dict[str, Any]:
    runner.run(
        "ElevationAboveStream",
        [f"--dem={cond_dem}", f"--streams={drainage}", f"--output={out_path}"],
        work_dir=work_dir, outputs=[out_path],
    )
    ensure_same_grid(grid, out_path, "HAND", warnings)
    hand, valid, _, _ = read_band(out_path)

    if (valid & ~dem_valid).any():
        raise ProcessingError("HAND has valid values on cells that are nodata in the conditioned DEM.")
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ProcessingError(
            "HAND raster has no valid cells (all nodata). Check the drainage threshold and the conditioned DEM."
        )
    values = hand[valid].astype(np.float64)
    if not np.isfinite(values).all():  # pragma: no cover - valid_mask already excludes these
        raise ProcessingError("HAND contains non-finite values.")

    on_drain = valid & (drain == 1)
    bad_drain = int((np.abs(hand[on_drain]) > 1e-6).sum())
    if bad_drain or (drain == 1).sum() != on_drain.sum():
        raise ProcessingError(
            f"Unexpected HAND on drainage cells: {bad_drain} non-zero, "
            f"{int((drain == 1).sum() - on_drain.sum())} nodata. HAND must be 0 on drainage cells."
        )
    n_unassoc = int((dem_valid & ~valid).sum())
    if n_unassoc:
        warnings.add(
            f"{n_unassoc} valid DEM cell(s) ({100 * n_unassoc / dem_valid.sum():.2f}%) have no associated drainage "
            "cell (their flow path leaves the extent or ends without reaching drainage); HAND is nodata there "
            "and they stay nodata in the final map."
        )
    n_neg = int((values < -1e-6).sum())
    if n_neg:
        warnings.add(f"{n_neg} cell(s) have negative HAND (terrain below its associated drainage cell).")

    pct = np.percentile(values, [5, 25, 50, 75, 95])
    stats = {
        "units": "vertical units of the DEM",
        "valid_cells": n_valid,
        "cells_without_drainage_association": n_unassoc,
        "negative_cells": n_neg,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(pct[0]), "p25": float(pct[1]), "median": float(pct[2]), "p75": float(pct[3]), "p95": float(pct[4]),
        "association": "nearest downslope drainage cell along the D8 flow path (ElevationAboveStream)",
    }
    log.info("HAND: min %.3f, median %.3f, max %.3f (DEM vertical units) over %d cells.",
             stats["min"], stats["median"], stats["max"], n_valid)
    return stats
