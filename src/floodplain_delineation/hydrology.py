"""Stages 3-4: D8 flow direction, flow accumulation and drainage-network extraction.

Flow routing is D8 throughout (single, consistent algorithm):
* ``D8Pointer`` -> flow direction (WhiteboxTools default, non-ESRI scheme:
  1=NE, 2=E, 4=SE, 8=S, 16=SW, 32=W, 64=NW, 128=N, 0=no downstream neighbour).
* ``D8FlowAccumulation`` (from that pointer, output type 'cells') -> contributing cells,
  where each cell counts itself (minimum value is 1).
* ``ExtractStreams`` -> drainage cells.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .exceptions import ProcessingError
from .logging_utils import WarningCollector, get_logger
from .rasters import GridInfo, ensure_same_grid, read_band, write_raster
from .spatial import CellGeometry
from .wbt import WhiteboxRunner

log = get_logger("hydrology")

D8_CODES = (0, 1, 2, 4, 8, 16, 32, 64, 128)
D8_CARDINAL_EW = (2, 32)
D8_CARDINAL_NS = (8, 128)
D8_DIAGONAL = (1, 4, 16, 64)

DRAINAGE_NODATA = 255
FLOAT32_EXACT_INT_LIMIT = 2 ** 24


def compute_flow_direction(
    runner: WhiteboxRunner, cond_dem: Path, out_path: Path, work_dir: Path,
    dem_valid: np.ndarray, grid: GridInfo, warnings: WarningCollector,
) -> Dict[str, Any]:
    runner.run("D8Pointer", [f"--dem={cond_dem}", f"--output={out_path}"], work_dir=work_dir, outputs=[out_path])
    ensure_same_grid(grid, out_path, "Flow direction", warnings)
    ptr, valid, _, _ = read_band(out_path)
    if (valid != dem_valid).any():
        raise ProcessingError(
            f"Flow direction nodata footprint differs from the conditioned DEM in {int((valid != dem_valid).sum())} cell(s)."
        )
    codes = np.unique(ptr[valid])
    unexpected = [int(c) for c in codes if int(c) not in D8_CODES]
    if unexpected:
        raise ProcessingError(f"Flow direction contains unexpected D8 codes: {unexpected}.")
    n_no_dir = int((ptr[valid] == 0).sum())
    log.info("Flow direction: %d valid cells, %d without a downstream neighbour (edges/outlets/unresolved flats).",
             int(valid.sum()), n_no_dir)
    return {"tool": "D8Pointer", "scheme": "WhiteboxTools default (non-ESRI)", "cells_without_direction": n_no_dir}


def compute_flow_accumulation(
    runner: WhiteboxRunner, fdir: Path, out_path: Path, work_dir: Path,
    dem_valid: np.ndarray, grid: GridInfo, warnings: WarningCollector,
) -> Dict[str, Any]:
    runner.run(
        "D8FlowAccumulation",
        [f"--input={fdir}", "--pntr", "--out_type=cells", f"--output={out_path}"],
        work_dir=work_dir, outputs=[out_path],
    )
    ensure_same_grid(grid, out_path, "Flow accumulation", warnings)
    acc, valid, _, _ = read_band(out_path)
    if (valid != dem_valid).any():
        raise ProcessingError(
            f"Flow accumulation nodata footprint differs from the conditioned DEM in {int((valid != dem_valid).sum())} cell(s)."
        )
    vals = acc[valid]
    if vals.min() < 1 or vals.max() > valid.sum() + 0.5:
        raise ProcessingError(
            f"Flow accumulation values out of range (min {vals.min()}, max {vals.max()}, valid cells {int(valid.sum())})."
        )
    if vals.max() > FLOAT32_EXACT_INT_LIMIT and acc.dtype == np.float32:
        warnings.add("Flow accumulation exceeds 2^24 cells stored as float32; cell counts lose integer precision.")
    return {"tool": "D8FlowAccumulation", "out_type": "cells", "max_accumulation_cells": float(vals.max())}


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
    effective_cells: int, dem_valid: np.ndarray, grid: GridInfo, warnings: WarningCollector,
) -> np.ndarray:
    """Extract drainage cells and store them as uint8 (1 drainage, 0 other, 255 nodata).

    WhiteboxTools ``ExtractStreams`` marks cells with accumulation STRICTLY greater than the
    threshold, so ``effective_cells - 0.5`` is passed to obtain ``accumulation >= effective_cells``
    (accumulation is integer-valued).
    """
    runner.run(
        "ExtractStreams",
        [f"--flow_accum={acc_path}", f"--threshold={effective_cells - 0.5}", "--zero_background", f"--output={raw_path}"],
        work_dir=work_dir, outputs=[raw_path],
    )
    ensure_same_grid(grid, raw_path, "Drainage network", warnings)
    raw, valid, _, _ = read_band(raw_path)
    if (valid != dem_valid).any():
        raise ProcessingError("Drainage raster nodata footprint differs from the conditioned DEM.")
    vals = np.unique(raw[valid])
    if not set(vals.tolist()) <= {0.0, 1.0}:
        raise ProcessingError(f"Unexpected values in drainage raster: {vals[:10].tolist()}.")
    drain = np.full(raw.shape, DRAINAGE_NODATA, dtype=np.uint8)
    drain[valid] = (raw[valid] == 1).astype(np.uint8)
    n = int((drain == 1).sum())
    if n == 0:
        acc, _, _, _ = read_band(acc_path)
        raise ProcessingError(
            f"No drainage cells were extracted with threshold {effective_cells} cells (maximum flow "
            f"accumulation in the processing extent: {acc[valid].max():.0f} cells). Lower "
            "processing.drainage.threshold_value or enlarge the buffer."
        )
    write_raster(
        out_path, drain, transform=grid.transform, crs=grid.crs, nodata=DRAINAGE_NODATA, compress=None,
        tags={"CLASS_VALUES": "1=drainage, 0=non-drainage, 255=nodata", "THRESHOLD_CELLS": effective_cells},
    )
    raw_path.unlink(missing_ok=True)
    return drain


def drainage_diagnostics(
    drain: np.ndarray, fdir_path: Path, cell: CellGeometry
) -> Dict[str, Any]:
    """Descriptive drainage statistics (never used to tune the threshold)."""
    ptr, _, _, _ = read_band(fdir_path)
    is_d = drain == 1
    n_valid = int((drain != DRAINAGE_NODATA).sum())
    p = ptr[is_d]
    n_ew = int(np.isin(p, D8_CARDINAL_EW).sum())
    n_ns = int(np.isin(p, D8_CARDINAL_NS).sum())
    n_dg = int(np.isin(p, D8_DIAGONAL).sum())
    length_m = n_ew * cell.dx_m + n_ns * cell.dy_m + n_dg * math.hypot(cell.dx_m, cell.dy_m)
    return {
        "drainage_cells": int(is_d.sum()),
        "valid_cells": n_valid,
        "percent_of_processing_area": 100.0 * int(is_d.sum()) / n_valid,
        "approx_length_km": length_m / 1000.0,
        "length_method": "sum of D8 downstream step lengths (cell centre to cell centre) over drainage cells",
    }
