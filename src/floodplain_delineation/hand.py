"""Stage 5: Height Above Nearest Drainage (HAND).

HAND(x) = Z(x) - Z_drain(x), with Z_drain the elevation of the nearest drainage cell
reached by following the D8 flow path downslope (WhiteboxTools ``ElevationAboveStream``).
Units are the vertical units of the DEM. Drainage cells have HAND = 0.
Cells that never reach a drainage cell (e.g. drain off the extent) are nodata.

Validation and statistics are computed block by block. Percentiles come from a fine histogram
(100 000 bins between the minimum and maximum HAND), i.e. they are accurate to one bin width,
which is reported in the summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio

from .exceptions import ProcessingError
from .hydrology import DRAINAGE_NODATA
from .logging_utils import WarningCollector, get_logger
from .rasters import GridInfo, ensure_same_grid, row_stripes, valid_mask
from .wbt import WhiteboxRunner

log = get_logger("hand")

HISTOGRAM_BINS = 100_000
PERCENTILES = (("p05", 0.05), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p95", 0.95))


def _histogram_quantile(counts: np.ndarray, lo: float, bin_width: float, q: float) -> float:
    total = int(counts.sum())
    cum = np.cumsum(counts)
    target = q * total
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(counts) - 1)
    before = int(cum[idx - 1]) if idx > 0 else 0
    frac = (target - before) / counts[idx] if counts[idx] else 0.0
    return float(lo + (idx + min(max(frac, 0.0), 1.0)) * bin_width)


def compute_hand(
    runner: WhiteboxRunner, cond_dem: Path, drainage: Path, out_path: Path, work_dir: Path,
    grid: GridInfo, warnings: WarningCollector,
) -> Dict[str, Any]:
    runner.run_raster(
        "ElevationAboveStream", [f"--dem={cond_dem}", f"--streams={drainage}"],
        work_dir=work_dir, output=out_path,
    )
    ensure_same_grid(grid, out_path, "HAND", warnings)

    # ---- pass 1: footprint checks, drainage consistency, count/min/max/mean/variance ----
    n_valid = n_dem_valid = n_extra = n_unassoc = n_neg = 0
    bad_drain = drain_cells = drain_nodata = 0
    lo, hi = np.inf, -np.inf
    mean, m2 = 0.0, 0.0
    with rasterio.open(cond_dem) as ref, rasterio.open(drainage) as dsrc, rasterio.open(out_path) as src:
        for win in row_stripes(src.height, src.width, 32):
            hand = src.read(1, window=win)
            valid = valid_mask(hand, src.nodata)
            dem_valid = valid_mask(ref.read(1, window=win), ref.nodata)
            drain = dsrc.read(1, window=win)
            n_dem_valid += int(dem_valid.sum())
            n_extra += int((valid & ~dem_valid).sum())
            n_unassoc += int((dem_valid & ~valid).sum())
            on_drain = drain == 1
            drain_cells += int(on_drain.sum())
            drain_nodata += int((on_drain & ~valid).sum())
            bad_drain += int((np.abs(hand[on_drain & valid].astype(np.float64)) > 1e-6).sum())
            v = hand[valid].astype(np.float64)
            if v.size:
                n_neg += int((v < -1e-6).sum())
                lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
                m_b = float(v.mean())
                m2_b = float(((v - m_b) ** 2).sum())
                tot = n_valid + v.size
                delta = m_b - mean
                mean += delta * v.size / tot
                m2 += m2_b + delta * delta * n_valid * v.size / tot
                n_valid = tot
    if n_extra:
        raise ProcessingError("HAND has valid values on cells that are nodata in the conditioned DEM.")
    if n_valid == 0:
        raise ProcessingError(
            "HAND raster has no valid cells (all nodata). Check the drainage threshold and the conditioned DEM."
        )
    if bad_drain or drain_nodata:
        raise ProcessingError(
            f"Unexpected HAND on drainage cells: {bad_drain} non-zero, "
            f"{drain_nodata} nodata. HAND must be 0 on drainage cells."
        )
    if n_unassoc:
        warnings.add(
            f"{n_unassoc} valid DEM cell(s) ({100 * n_unassoc / n_dem_valid:.2f}%) have no associated drainage "
            "cell (their flow path leaves the extent or ends without reaching drainage); HAND is nodata there "
            "and they stay nodata in the final map."
        )
    if n_neg:
        warnings.add(f"{n_neg} cell(s) have negative HAND (terrain below its associated drainage cell).")

    # ---- pass 2: histogram for percentiles ----
    counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    bin_width = (hi - lo) / HISTOGRAM_BINS if hi > lo else 0.0
    if hi > lo:
        with rasterio.open(out_path) as src:
            for win in row_stripes(src.height, src.width, 16):
                hand = src.read(1, window=win)
                v = hand[valid_mask(hand, src.nodata)].astype(np.float64)
                counts += np.histogram(v, bins=HISTOGRAM_BINS, range=(lo, hi))[0]
        pct = {name: _histogram_quantile(counts, lo, bin_width, q) for name, q in PERCENTILES}
    else:
        pct = {name: float(lo) for name, _ in PERCENTILES}

    stats = {
        "units": "vertical units of the DEM",
        "valid_cells": n_valid,
        "cells_without_drainage_association": n_unassoc,
        "negative_cells": n_neg,
        "min": float(lo),
        "max": float(hi),
        "mean": float(mean),
        "std": float(np.sqrt(m2 / n_valid)),
        **pct,
        "percentile_method": f"histogram with {HISTOGRAM_BINS} bins (bin width {bin_width:.6g})",
        "association": "nearest downslope drainage cell along the D8 flow path (ElevationAboveStream)",
    }
    log.info("HAND: min %.3f, median %.3f, max %.3f (DEM vertical units) over %d cells.",
             stats["min"], stats["median"], stats["max"], n_valid)
    return stats
