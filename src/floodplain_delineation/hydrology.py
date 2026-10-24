"""Stages 3-4: D8 flow direction, flow accumulation and drainage-network extraction.

Flow routing is D8 throughout (single, consistent algorithm):
* ``D8Pointer`` -> flow direction (WhiteboxTools default, non-ESRI scheme:
  1=NE, 2=E, 4=SE, 8=S, 16=SW, 32=W, 64=NW, 128=N, 0=no downstream neighbour).
* ``D8FlowAccumulation`` (from that pointer, output type 'cells') -> contributing cells,
  where each cell counts itself (minimum value is 1).
* ``ExtractStreams`` -> drainage cells.

Validations read the rasters block by block (full-width row stripes) and compare their nodata
footprint with the conditioned DEM, which is passed as a *path* (``reference``), so memory use
does not depend on the raster size.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio

from .exceptions import ProcessingError
from .logging_utils import WarningCollector, get_logger
from .rasters import GridInfo, create_raster, ensure_same_grid, row_stripes, scan_min_max, valid_mask
from .spatial import CellGeometry
from .wbt import WhiteboxRunner

log = get_logger("hydrology")

D8_CODES = (0, 1, 2, 4, 8, 16, 32, 64, 128)
D8_CARDINAL_EW = (2, 32)
D8_CARDINAL_NS = (8, 128)
D8_DIAGONAL = (1, 4, 16, 64)

DRAINAGE_NODATA = 255
FLOAT32_EXACT_INT_LIMIT = 2 ** 24
#: Relative tolerance for accumulation sanity checks: float32 cannot store cell counts above
#: 2^24 exactly (relative error <= 6e-8), so a strict ``max <= valid_cells`` test could fail by rounding.
ACCUMULATION_RTOL = 1e-6


def compute_flow_direction(
    runner: WhiteboxRunner, cond_dem: Path, out_path: Path, work_dir: Path,
    grid: GridInfo, warnings: WarningCollector,
) -> Dict[str, Any]:
    runner.run_raster("D8Pointer", [f"--dem={cond_dem}"], work_dir=work_dir, output=out_path)
    ensure_same_grid(grid, out_path, "Flow direction", warnings)
    n_mismatch = n_valid = n_no_dir = 0
    codes = set()
    with rasterio.open(cond_dem) as ref, rasterio.open(out_path) as src:
        for win in row_stripes(src.height, src.width, 16):
            ptr = src.read(1, window=win)
            valid = valid_mask(ptr, src.nodata)
            n_mismatch += int((valid != valid_mask(ref.read(1, window=win), ref.nodata)).sum())
            v = ptr[valid]
            n_valid += int(v.size)
            n_no_dir += int((v == 0).sum())
            codes.update(int(c) for c in np.unique(v))
    if n_mismatch:
        raise ProcessingError(
            f"Flow direction nodata footprint differs from the conditioned DEM in {n_mismatch} cell(s)."
        )
    unexpected = sorted(c for c in codes if c not in D8_CODES)
    if unexpected:
        raise ProcessingError(f"Flow direction contains unexpected D8 codes: {unexpected}.")
    log.info("Flow direction: %d valid cells, %d without a downstream neighbour (edges/outlets/unresolved flats).",
             n_valid, n_no_dir)
    return {"tool": "D8Pointer", "scheme": "WhiteboxTools default (non-ESRI)", "cells_without_direction": n_no_dir}


def compute_flow_accumulation(
    runner: WhiteboxRunner, fdir: Path, out_path: Path, work_dir: Path,
    reference: Path, grid: GridInfo, warnings: WarningCollector,
) -> Dict[str, Any]:
    runner.run_raster(
        "D8FlowAccumulation", [f"--input={fdir}", "--pntr", "--out_type=cells"],
        work_dir=work_dir, output=out_path,
    )
    ensure_same_grid(grid, out_path, "Flow accumulation", warnings)
    n_mismatch = n_valid = 0
    lo = hi = None
    with rasterio.open(reference) as ref, rasterio.open(out_path) as src:
        dtype = src.dtypes[0]
        for win in row_stripes(src.height, src.width, 16):
            acc = src.read(1, window=win)
            valid = valid_mask(acc, src.nodata)
            n_mismatch += int((valid != valid_mask(ref.read(1, window=win), ref.nodata)).sum())
            v = acc[valid]
            if v.size:
                n_valid += int(v.size)
                lo = float(v.min()) if lo is None else min(lo, float(v.min()))
                hi = float(v.max()) if hi is None else max(hi, float(v.max()))
    if n_mismatch:
        raise ProcessingError(
            f"Flow accumulation nodata footprint differs from the conditioned DEM in {n_mismatch} cell(s)."
        )
    if lo is None or lo < 1 - ACCUMULATION_RTOL or hi > n_valid * (1 + ACCUMULATION_RTOL) + 1:
        raise ProcessingError(
            f"Flow accumulation values out of range (min {lo}, max {hi}, valid cells {n_valid})."
        )
    if hi > FLOAT32_EXACT_INT_LIMIT and dtype == "float32":
        warnings.add("Flow accumulation exceeds 2^24 cells stored as float32; cell counts lose integer precision.")
    return {"tool": "D8FlowAccumulation", "out_type": "cells", "max_accumulation_cells": float(hi)}


def resolve_drainage_threshold(
    threshold_type: str, value: float, cell: CellGeometry, warnings: WarningCollector
) -> Dict[str, Any]:
    """Convert the configured threshold into a minimum number of contributing cells.

    Drainage cells are those with accumulation >= ``effective_cells``.
    """
    if threshold_type == "cells":
        cells = int(value)
        info = {"type": "cells", "configured_value": value}
    else:
        if cell.geographic:
            var = f"{cell.area_variation_fraction:.2%}" if cell.area_variation_fraction is not None else "unknown"
            warnings.add(
                "Area-based drainage threshold with a geographic CRS: cell area varies with latitude. Using the "
                f"geodesic cell area at latitude {cell.reference_latitude:.3f} ({cell.area_km2:.6f} km2; area "
                f"varies by {var} across the extent). A projected CRS is recommended."
            )
        if cell.area_m2 <= 0:
            raise ProcessingError("Cell area is not positive; cannot convert km2 threshold to cells.")
        cells = max(1, math.ceil(value * 1e6 / cell.area_m2 - 1e-9))
        info = {"type": "area_km2", "configured_value": value}
    info.update(
        effective_cells=cells,
        cell_area_km2=cell.area_km2,
        effective_area_km2=cells * cell.area_km2,
        rule="drainage cell if contributing cells >= effective_cells",
    )
    return info


def extract_drainage(
    runner: WhiteboxRunner, acc_path: Path, out_path: Path, raw_path: Path, work_dir: Path,
    effective_cells: int, reference: Path, grid: GridInfo, warnings: WarningCollector,
) -> Dict[str, Any]:
    """Extract drainage cells and store them as uint8 (1 drainage, 0 other, 255 nodata).

    WhiteboxTools ``ExtractStreams`` marks cells with accumulation STRICTLY greater than the
    threshold, so ``effective_cells - 0.5`` is passed to obtain ``accumulation >= effective_cells``
    (accumulation is integer-valued).
    """
    runner.run_raster(
        "ExtractStreams",
        [f"--flow_accum={acc_path}", f"--threshold={effective_cells - 0.5}", "--zero_background"],
        work_dir=work_dir, output=raw_path, normalize=False,  # read only by Python, no tool reads it afterwards
    )
    try:
        ensure_same_grid(grid, raw_path, "Drainage network", warnings)
        n_mismatch = n_drain = 0
        bad_values = set()
        with rasterio.open(reference) as ref, rasterio.open(raw_path) as src, create_raster(
            out_path, height=src.height, width=src.width, dtype="uint8", transform=grid.transform,
            crs=grid.crs, nodata=DRAINAGE_NODATA,
            tags={"CLASS_VALUES": "1=drainage, 0=non-drainage, 255=nodata", "THRESHOLD_CELLS": effective_cells},
        ) as dst:
            for win in row_stripes(src.height, src.width, 16):
                raw = src.read(1, window=win)
                valid = valid_mask(raw, src.nodata)
                n_mismatch += int((valid != valid_mask(ref.read(1, window=win), ref.nodata)).sum())
                bad_values.update(v for v in np.unique(raw[valid]).tolist() if v not in (0.0, 1.0))
                drain = np.full(raw.shape, DRAINAGE_NODATA, dtype=np.uint8)
                drain[valid] = (raw[valid] == 1).astype(np.uint8)
                n_drain += int((drain == 1).sum())
                dst.write(drain, 1, window=win)
            if n_mismatch:
                raise ProcessingError("Drainage raster nodata footprint differs from the conditioned DEM.")
            if bad_values:
                raise ProcessingError(f"Unexpected values in drainage raster: {sorted(bad_values)[:10]}.")
            if n_drain == 0:
                _, acc_max, _ = scan_min_max(acc_path)
                raise ProcessingError(
                    f"No drainage cells were extracted with threshold {effective_cells} cells (maximum flow "
                    f"accumulation in the processing extent: {acc_max:.0f} cells). Lower "
                    "processing.drainage.threshold_value or enlarge the buffer."
                )
    finally:
        raw_path.unlink(missing_ok=True)
    return {"drainage_cells": n_drain}


def drainage_diagnostics(drain_path: Path, fdir_path: Path, cell: CellGeometry) -> Dict[str, Any]:
    """Descriptive drainage statistics (never used to tune the threshold)."""
    n_valid = n_d = n_ew = n_ns = n_dg = 0
    with rasterio.open(drain_path) as dsrc, rasterio.open(fdir_path) as psrc:
        for win in row_stripes(dsrc.height, dsrc.width, 8):
            drain = dsrc.read(1, window=win)
            is_d = drain == 1
            n_valid += int((drain != DRAINAGE_NODATA).sum())
            n_d += int(is_d.sum())
            p = psrc.read(1, window=win)[is_d]
            n_ew += int(np.isin(p, D8_CARDINAL_EW).sum())
            n_ns += int(np.isin(p, D8_CARDINAL_NS).sum())
            n_dg += int(np.isin(p, D8_DIAGONAL).sum())
    length_m = n_ew * cell.dx_m + n_ns * cell.dy_m + n_dg * math.hypot(cell.dx_m, cell.dy_m)
    return {
        "drainage_cells": n_d,
        "valid_cells": n_valid,
        "percent_of_processing_area": 100.0 * n_d / n_valid,
        "approx_length_km": length_m / 1000.0,
        "length_method": "sum of D8 downstream step lengths (cell centre to cell centre) over drainage cells",
    }
