"""Relative Elevation Model (REM) by IDW interpolation of drainage elevations, independently per basin.

    REM(x, y) = DEM(x, y) - Z_drainage_IDW_basin(x, y)

HAND associates each cell with the drainage cell reached by its D8 path (existing algorithm). The REM is a
different product: it subtracts a surface interpolated *euclidean-ly* from elevations sampled on the drainage
network, using only the samples of the basin that contains the cell. It is not an observed or hydraulic water
surface, and IDW neither guarantees longitudinal hydrological continuity nor respects topographic barriers
inside a basin.

Stages (all raster work is done on full-width row stripes; no whole-raster array is ever held):

1. :func:`delineate_basins`   WhiteboxTools ``Subbasins`` on the existing D8 pointer and drainage network: one
   exclusive (non-nested) catchment per stream link; validated IDs raster.
2. :func:`sample_drainage`    systematic sampling of drainage cells (one per N x N block and basin).
3. :func:`interpolate_surface` exact IDW per basin (all samples of the basin, no search radius).
4. :func:`compute_rem`        DEM minus the surface.

Reference DEM: the unconditioned DEM on the master grid (``dem_processing_crs.tif``). Breaching/filling alter
elevations along the flow paths, so the observed topography is sampled instead (CIDI section 2, item 2).

Basin without any drainage sample: its cells are NoData in the surface and in the REM, and the basin and cell
counts are recorded as a warning. No value is invented and the run does not fail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from .exceptions import ProcessingError
from .hand import HISTOGRAM_BINS, _histogram_quantile
from .logging_utils import WarningCollector, get_logger
from .rasters import (
    GridInfo,
    create_raster,
    ensure_same_grid,
    row_stripes,
    shift_rows,
    valid_mask,
)
from .wbt import WhiteboxRunner

log = get_logger("rem")

BASINS_TOOL = "Subbasins"
BASINS_FLAGS = ["--d8_pntr", "--streams"]  # flags passed (besides --output); see the help of the installed version
STREAMS_INPUT_NODATA = -9999               # must not be a possible basin ID (the tool copies it to its output)
SAMPLING_METHOD_ID = "block_nearest_centre_per_basin_v1"
IDW_METHOD_ID = "idw_all_basin_samples_euclidean_v1"
REM_METHOD_ID = "dem_minus_idw_surface_v1"

FLOAT32_EXACT_INT_LIMIT = 2 ** 24
TILE = 512
#: Distance pairs (target cells x samples) evaluated per chunk; bounds the IDW working memory
#: (about 6 float64 arrays of this size) without ever reducing the set of samples used.
IDW_PAIR_BUDGET = 4_000_000
PROGRESS_SECONDS = 60.0

BASINS_SEMANTICS = (
    "WhiteboxTools 'Subbasins' (help of the installed version: 'identifies the catchments, or sub-basin, draining to "
    "each link in a stream network'; inputs: D8 pointer + streams raster). Behaviour verified on synthetic data with "
    "the installed version: one unit per stream link (stream segment between confluences/heads, derived from the "
    "pipeline's drainage network); every cell that drains to the network belongs to exactly ONE unit, units are not "
    "nested; the drainage cells of a link carry its ID; IDs are positive integers but NOT contiguous; cells whose D8 "
    "path leaves the extent without reaching the network have no ID (NoData). No outlet points are needed or invented. "
    "The units are as fine or as coarse as the drainage threshold makes the network (a large area threshold gives "
    "few, large units)."
)


# --------------------------------------------------------------------------- #
# REM-1: basins
# --------------------------------------------------------------------------- #
def _write_streams_input(drain_path: Path, out_path: Path) -> None:
    """int32 copy of the drainage raster for ``Subbasins``: 1 drainage, 0 other, NoData -9999.

    The tool gives its output the NoData of the streams raster. The pipeline's own drainage raster is uint8
    with NoData 255, which would turn a legitimate basin ID 255 into NoData; -9999 can never be an ID.
    """
    with rasterio.open(drain_path) as src, create_raster(
        out_path, height=src.height, width=src.width, dtype="int32", transform=src.transform, crs=src.crs,
        nodata=STREAMS_INPUT_NODATA,
    ) as dst:
        for win in row_stripes(src.height, src.width, 12):
            d = src.read(1, window=win)
            out = np.where(valid_mask(d, src.nodata), (d == 1).astype(np.int32), np.int32(STREAMS_INPUT_NODATA))
            dst.write(out.astype(np.int32), 1, window=win)


def _add_counts(counts: Dict[int, int], values: np.ndarray) -> None:
    ids, n = np.unique(values, return_counts=True)
    for i, c in zip(ids.tolist(), n.tolist()):
        counts[i] = counts.get(i, 0) + c


def delineate_basins(
    runner: WhiteboxRunner, fdir_path: Path, drain_path: Path, out_path: Path, work_dir: Path, grid: GridInfo,
    warnings: WarningCollector,
) -> Dict[str, Any]:
    """Run ``Subbasins`` (D8 pointer + drainage network), normalise the output with GDAL and validate it."""
    streams_in = work_dir / "rem_streams_input.tif"
    _write_streams_input(drain_path, streams_in)
    try:
        runner.run_raster(
            BASINS_TOOL, [f"--d8_pntr={fdir_path}", f"--streams={streams_in}"], work_dir=work_dir, output=out_path
        )
    finally:
        streams_in.unlink(missing_ok=True)
    ensure_same_grid(grid, out_path, "Basin IDs", warnings)

    sizes: Dict[int, int] = {}
    drain_cells_per_basin: Dict[int, int] = {}
    n_ptr_valid = n_mismatch = n_non_integer = n_nonpositive = n_drain_without_id = 0
    with rasterio.open(fdir_path) as ref, rasterio.open(drain_path) as dr, rasterio.open(out_path) as src:
        dtype = np.dtype(src.dtypes[0])
        floating = np.issubdtype(dtype, np.floating)
        nodata = src.nodata
        if nodata is not None and nodata >= 1:
            raise ProcessingError(
                f"The basin raster declares NoData {nodata:g}, which is also a possible basin ID; the units could not be "
                "told from missing values. Aborting the REM."
            )
        for win in row_stripes(src.height, src.width, 24):
            ids = src.read(1, window=win)
            has_value = valid_mask(ids, nodata)
            ptr_valid = valid_mask(ref.read(1, window=win), ref.nodata)
            drain = dr.read(1, window=win) == 1
            n_ptr_valid += int(ptr_valid.sum())
            n_mismatch += int((has_value & ~ptr_valid).sum())          # an ID on a cell that is not DEM data
            v = ids[has_value & ptr_valid]
            if floating:
                n_non_integer += int((v != np.floor(v)).sum())
            n_nonpositive += int((v < 1).sum())
            iv = v[v >= 1].astype(np.int64)
            if iv.size:
                _add_counts(sizes, iv)
            has_id = has_value & ptr_valid & (ids >= 1)
            n_drain_without_id += int((drain & ~has_id).sum())
            dv = ids[drain & has_id].astype(np.int64)
            if dv.size:
                _add_counts(drain_cells_per_basin, dv)
    if not sizes:
        raise ProcessingError(
            "Subbasins produced no valid basin units (all cells without an ID). The REM cannot be computed; no "
            "artificial subdivision is made."
        )
    if n_mismatch:
        raise ProcessingError(
            f"The basin IDs raster has {n_mismatch} ID value(s) on cells that are NoData in the D8 pointer; the "
            "semantics of the installed Subbasins tool do not match the expected one. Aborting the REM."
        )
    if n_non_integer or n_nonpositive:
        raise ProcessingError(
            f"Basin IDs must be positive integers ({n_non_integer} non-integer and {n_nonpositive} non-positive values found)."
        )
    if n_drain_without_id:
        raise ProcessingError(
            f"{n_drain_without_id} drainage cell(s) have no basin ID; every stream cell must belong to the unit of its "
            "link. The semantics of the installed Subbasins tool do not match the expected one. Aborting the REM."
        )
    if dtype == np.float32 and max(sizes) > FLOAT32_EXACT_INT_LIMIT:
        raise ProcessingError(
            f"The largest basin ID ({max(sizes)}) exceeds 2^24 and cannot be stored exactly as float32 by this "
            "WhiteboxTools version."
        )
    cells = np.array(sorted(sizes.values()), dtype=np.int64)
    n_basins = int(cells.size)
    n_with_id = int(cells.sum())
    n_without_id = n_ptr_valid - n_with_id
    no_drain = n_basins - len(drain_cells_per_basin)
    if no_drain:
        warnings.add(
            f"{no_drain} basin unit(s) contain no drainage cell; they cannot receive IDW samples and become NoData in the REM."
        )
    if n_basins == 1:
        log.info("Subbasins produced a single unit covering the drained area; it is used as is.")
    per = np.array(sorted(drain_cells_per_basin.values()), dtype=np.int64)
    stats = {
        "tool": BASINS_TOOL,
        "flags_passed": BASINS_FLAGS,
        "inputs": "flow_direction (D8 pointer of the conditioned DEM) and drainage_network (as an int32 raster, NoData -9999)",
        "semantics": BASINS_SEMANTICS,
        "dtype": dtype.name,
        "nodata": nodata,
        "cells_with_id": n_with_id,
        "cells_without_id": n_without_id,
        "cells_without_id_fraction": n_without_id / n_ptr_valid if n_ptr_valid else 0.0,
        "cells_without_id_meaning": "valid DEM cells whose D8 path does not reach the drainage network inside the extent",
        "basins": n_basins,
        "largest_id": max(sizes),
        "ids_contiguous": bool(max(sizes) == n_basins),
        "basin_cells_min": int(cells.min()),
        "basin_cells_median": float(np.median(cells)),
        "basin_cells_p95": float(np.percentile(cells, 95)),
        "basin_cells_max": int(cells.max()),
        "largest_basin_fraction": float(cells.max()) / n_with_id,
        "basins_smaller_than_10_cells": int((cells < 10).sum()),
        "basins_without_drainage_cells": no_drain,
        "drainage_cells_per_basin_min": int(per.min()) if per.size else 0,
        "drainage_cells_per_basin_median": float(np.median(per)) if per.size else 0.0,
        "drainage_cells_per_basin_max": int(per.max()) if per.size else 0,
        "depends_on": "conditioned DEM, D8 pointer, drainage network (threshold), extent and buffer of the processing area",
    }
    log.info(
        "Subbasins: %d units (one per stream link); cells per unit min %d / median %.0f / max %d (%.1f%% of the drained "
        "cells in the largest); %d with < 10 cells; %d cell(s) (%.2f%%) drain to no stream and have no unit.",
        n_basins, stats["basin_cells_min"], stats["basin_cells_median"], stats["basin_cells_max"],
        100 * stats["largest_basin_fraction"], stats["basins_smaller_than_10_cells"], n_without_id,
        100 * stats["cells_without_id_fraction"],
    )
    return stats


# --------------------------------------------------------------------------- #
# REM-2: sampling
# --------------------------------------------------------------------------- #
@dataclass
class SampleTable:
    """Drainage samples sorted by basin ID (then row, column)."""

    basin: np.ndarray   # int64
    row: np.ndarray     # int64 (master grid)
    col: np.ndarray     # int64
    x: np.ndarray       # float64, cell centre, processing CRS
    y: np.ndarray
    z: np.ndarray       # float64, reference-DEM elevation

    def __len__(self) -> int:
        return int(self.basin.size)

    def index(self) -> Dict[int, Tuple[int, int]]:
        """basin ID -> [start, end) slice of the (sorted) arrays."""
        if not len(self):
            return {}
        ids, start, cnt = np.unique(self.basin, return_index=True, return_counts=True)
        return {int(i): (int(a), int(a + c)) for i, a, c in zip(ids, start, cnt)}

    def counts_per_basin(self) -> np.ndarray:
        return np.unique(self.basin, return_counts=True)[1] if len(self) else np.zeros(0, dtype=np.int64)


def save_samples(path: Path, table: SampleTable, meta: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, basin=table.basin, row=table.row, col=table.col, x=table.x, y=table.y, z=table.z,
            meta=np.array([repr(meta)]),
        )
    tmp.replace(path)


def load_samples(path: Path) -> SampleTable:
    try:
        with np.load(path, allow_pickle=False) as d:
            return SampleTable(d["basin"], d["row"], d["col"], d["x"], d["y"], d["z"])
    except Exception as exc:
        raise ProcessingError(f"Cannot read the REM sample table {path}: {exc}") from exc


def select_block_samples(
    rows: np.ndarray, cols: np.ndarray, basin: np.ndarray, interval: int, res_x: float, res_y: float
) -> np.ndarray:
    """Indices of the selected samples among candidate cells (absolute master-grid ``rows``/``cols``).

    Rule: the grid is tiled with ``interval x interval`` pixel blocks anchored at row 0, column 0. For each
    (block, basin) pair the candidate nearest to the block centre is kept (distance in CRS units; ties broken
    by smaller row, then smaller column). The result depends only on the cell positions, never on the order
    in which they are read.
    """
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    br, bc = rows // interval, cols // interval
    cr = br * interval + (interval - 1) / 2.0
    cc = bc * interval + (interval - 1) / 2.0
    d2 = ((cols - cc) * res_x) ** 2 + ((rows - cr) * res_y) ** 2
    order = np.lexsort((cols, rows, d2, basin, bc, br))  # primary key: br, then bc, basin, distance, row, col
    b0, c0, i0 = br[order], bc[order], basin[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = (b0[1:] != b0[:-1]) | (c0[1:] != c0[:-1]) | (i0[1:] != i0[:-1])
    return order[first]


def sample_drainage(
    drain_path: Path, dem_path: Path, basins_path: Path, interval: int
) -> Tuple[SampleTable, Dict[str, Any]]:
    """Systematic sampling of valid drainage cells; each sample keeps its basin ID and DEM elevation."""
    parts: List[Tuple[np.ndarray, ...]] = []
    n_candidates = 0
    candidate_basins: List[np.ndarray] = []
    with rasterio.open(drain_path) as dr, rasterio.open(dem_path) as dem, rasterio.open(basins_path) as bs:
        t = dem.transform
        for win in row_stripes(dem.height, dem.width, 40, align=interval):
            r0 = int(win.row_off)
            drain = dr.read(1, window=win)
            z = dem.read(1, window=win)
            ids = bs.read(1, window=win)
            cand = (drain == 1) & valid_mask(z, dem.nodata) & valid_mask(ids, bs.nodata) & (ids >= 1)
            rr, cc = np.nonzero(cand)
            if rr.size == 0:
                continue
            n_candidates += int(rr.size)
            basin = ids[rr, cc].astype(np.int64)
            candidate_basins.append(np.unique(basin))
            rows = rr.astype(np.int64) + r0
            cols = cc.astype(np.int64)
            keep = select_block_samples(rows, cols, basin, interval, abs(t.a), abs(t.e))
            parts.append((basin[keep], rows[keep], cols[keep], z[rr, cc][keep].astype(np.float64)))
        transform = t
    if parts:
        basin = np.concatenate([p[0] for p in parts])
        rows = np.concatenate([p[1] for p in parts])
        cols = np.concatenate([p[2] for p in parts])
        z = np.concatenate([p[3] for p in parts])
    else:
        basin = rows = cols = np.zeros(0, dtype=np.int64)
        z = np.zeros(0, dtype=np.float64)
    order = np.lexsort((cols, rows, basin))
    basin, rows, cols, z = basin[order], rows[order], cols[order], z[order]
    table = SampleTable(
        basin=basin, row=rows, col=cols,
        x=transform.c + (cols + 0.5) * transform.a, y=transform.f + (rows + 0.5) * transform.e, z=z,
    )
    per_basin = table.counts_per_basin()
    net_basins = int(np.unique(np.concatenate(candidate_basins)).size) if candidate_basins else 0
    stats = {
        "method": SAMPLING_METHOD_ID,
        "rule": (
            f"grid tiled in {interval}x{interval}-pixel blocks anchored at row 0/column 0; for each block and basin "
            "the valid drainage cell (valid reference-DEM elevation, valid basin ID) nearest to the block centre is "
            "kept; ties go to the smaller row, then the smaller column"
        ),
        "interval_pixels": interval,
        "candidate_drainage_cells": n_candidates,
        "samples": len(table),
        "basins_intersecting_network": net_basins,
        "basins_with_samples": int(per_basin.size),
        "samples_per_basin_min": int(per_basin.min()) if per_basin.size else 0,
        "samples_per_basin_median": float(np.median(per_basin)) if per_basin.size else 0.0,
        "samples_per_basin_max": int(per_basin.max()) if per_basin.size else 0,
    }
    if stats["basins_with_samples"] != net_basins:  # cannot happen with the per-(block, basin) rule
        raise ProcessingError("Internal error: a basin that intersects the drainage network received no sample.")
    log.info(
        "REM samples: %d from %d candidate drainage cells (interval %d px); %d basin(s) with samples "
        "(per basin: min %d / median %.0f / max %d).",
        stats["samples"], n_candidates, interval, stats["basins_with_samples"],
        stats["samples_per_basin_min"], stats["samples_per_basin_median"], stats["samples_per_basin_max"],
    )
    return table, stats


# --------------------------------------------------------------------------- #
# REM-3: IDW per basin
# --------------------------------------------------------------------------- #
def idw_estimate(
    xs: np.ndarray, ys: np.ndarray, sx: np.ndarray, sy: np.ndarray, sz: np.ndarray, power: float
) -> np.ndarray:
    """IDW estimate at target points from ALL the given samples (euclidean distance, ``w = 1 / d^power``).

    Evaluated in chunks of target points so that at most ``IDW_PAIR_BUDGET`` distances are held at once; every
    sample takes part in every estimate. A target coinciding with a sample receives exactly its value. Weights
    are computed relative to the nearest sample, ``(d_min / d)^power`` (mathematically identical after
    normalisation), which cannot underflow.
    """
    n, m = sx.size, xs.size
    if n == 0:
        raise ValueError("IDW needs at least one sample")
    out = np.empty(m, dtype=np.float64)
    step = max(1, IDW_PAIR_BUDGET // n)
    half = power / 2.0
    for a in range(0, m, step):
        b = min(m, a + step)
        dx = xs[a:b, None] - sx[None, :]
        dy = ys[a:b, None] - sy[None, :]
        d2 = dx * dx + dy * dy
        del dx, dy
        nearest = d2.argmin(axis=1)
        d2min = d2[np.arange(b - a), nearest]
        coincident = d2min == 0.0
        safe_min = np.where(coincident, 1.0, d2min)
        d2 = np.where(d2 == 0.0, 1.0, d2)       # coincident rows are overwritten below
        ratio = safe_min[:, None] / d2
        w = ratio if power == 2.0 else np.power(ratio, half)
        est = (w @ sz) / w.sum(axis=1)
        est[coincident] = sz[nearest[coincident]]
        out[a:b] = est
    return out


def _creation_options() -> Dict[str, Any]:
    return {"predictor": 3}  # floating-point predictor for DEFLATE


def interpolate_surface(
    samples: SampleTable, basins_path: Path, dem_path: Path, out_path: Path, grid: GridInfo, power: float,
    warnings: WarningCollector,
) -> Dict[str, Any]:
    """Write the IDW surface (float32): each basin only from its own samples; NoData elsewhere."""
    index = samples.index()
    started = time.monotonic()
    last_log = started
    n_target = n_nodata_basin = n_nonfinite = n_pairs = 0
    basins_no_samples: set = set()
    basins_done: set = set()
    with rasterio.open(dem_path) as dem, rasterio.open(basins_path) as bs:
        out_nodata = dem.nodata
        total_rows = dem.height
        with create_raster(
            out_path, height=dem.height, width=dem.width, dtype="float32", transform=dem.transform, crs=dem.crs,
            nodata=out_nodata, compress="deflate", tiled=dem.height >= TILE and dem.width >= TILE,
            creation_options=_creation_options(),
            tags={"PRODUCT": "IDW surface of drainage elevations, per basin", "IDW_POWER": power,
                  "METHOD": IDW_METHOD_ID},
            description="rem_drainage_surface",
        ) as dst:
            tiled = dem.height >= TILE and dem.width >= TILE
            for win in row_stripes(dem.height, dem.width, 64, align=TILE if tiled else 1):
                r0, rows, width = int(win.row_off), int(win.height), int(win.width)
                z = dem.read(1, window=win)
                ids = bs.read(1, window=win)
                valid = valid_mask(z, dem.nodata) & valid_mask(ids, bs.nodata) & (ids >= 1)
                surf = np.full((rows, width), out_nodata, dtype=np.float32)
                flat = np.flatnonzero(valid.ravel())
                if flat.size:
                    bflat = ids.ravel()[flat].astype(np.int64)
                    order = np.argsort(bflat, kind="stable")
                    sorted_ids = bflat[order]
                    cuts = np.flatnonzero(np.diff(sorted_ids)) + 1
                    starts = np.concatenate(([0], cuts))
                    ends = np.concatenate((cuts, [sorted_ids.size]))
                    t = shift_rows(dem.transform, r0)
                    for a, b in zip(starts, ends):
                        bid = int(sorted_ids[a])
                        sl = index.get(bid)
                        pos = flat[order[a:b]]
                        n_target += int(pos.size)
                        if sl is None:
                            basins_no_samples.add(bid)
                            n_nodata_basin += int(pos.size)
                            continue
                        n_pairs += int(pos.size) * int(sl[1] - sl[0])
                        rr, cc = pos // width, pos % width
                        xs = t.c + (cc + 0.5) * t.a
                        ys = t.f + (rr + 0.5) * t.e
                        est = idw_estimate(xs, ys, samples.x[sl[0]:sl[1]], samples.y[sl[0]:sl[1]],
                                           samples.z[sl[0]:sl[1]], power)
                        surf[rr, cc] = est.astype(np.float32)
                        basins_done.add(bid)
                        n_nonfinite += int((~np.isfinite(est)).sum())
                if n_nonfinite:
                    raise ProcessingError("The IDW surface contains non-finite values; aborting.")
                dst.write(surf, 1, window=win)
                now = time.monotonic()
                if now - last_log >= PROGRESS_SECONDS:
                    done = r0 + rows
                    eta = (now - started) / done * (total_rows - done)
                    log.info("IDW surface: %d/%d rows (%.1f%%), %.0f s elapsed, ~%.0f s remaining.",
                             done, total_rows, 100 * done / total_rows, now - started, eta)
                    last_log = now
    elapsed = time.monotonic() - started
    if basins_no_samples:
        warnings.add(
            f"{len(basins_no_samples)} basin(s) containing {n_nodata_basin} valid DEM cell(s) have no drainage sample "
            "(no drainage cell inside them); their cells are NoData in the REM surface and in the REM."
        )
    ensure_same_grid(grid, out_path, "REM drainage surface", warnings)
    stats = {
        "method": IDW_METHOD_ID,
        "equation": "Z(x) = sum(w_i z_i) / sum(w_i), w_i = 1 / d(x, x_i)^p, euclidean d in processing-CRS units",
        "idw_power": power,
        "samples_used": "all samples of the basin (no search radius, neighbour limit or truncation)",
        "target_cells": n_target,
        "interpolated_cells": n_target - n_nodata_basin,
        "basins_interpolated": len(basins_done),
        "basins_without_samples": len(basins_no_samples),
        "cells_in_basins_without_samples": n_nodata_basin,
        "policy_basin_without_samples": "NoData in surface and REM; counted in the summary",
        "pair_budget_per_chunk": IDW_PAIR_BUDGET,
        "pair_evaluations": n_pairs,
        "pair_evaluations_per_second": round(n_pairs / elapsed) if elapsed > 0 else None,
        "cost_note": "cost = sum over basins of (cells in the basin x samples in the basin); measured, not extrapolated",
        "seconds": round(elapsed, 2),
        "dtype": "float32",
        "nodata": out_nodata,
    }
    log.info("IDW surface: %d cells in %d basin(s) interpolated in %.1f s (%.2e cell-sample evaluations, %.2e/s); "
             "%d cell(s) in %d basin(s) without samples.",
             stats["interpolated_cells"], stats["basins_interpolated"], elapsed, n_pairs,
             stats["pair_evaluations_per_second"] or 0, n_nodata_basin, len(basins_no_samples))
    return stats


# --------------------------------------------------------------------------- #
# REM-4: DEM - surface
# --------------------------------------------------------------------------- #
def compute_rem(
    dem_path: Path, surface_path: Path, out_path: Path, grid: GridInfo, warnings: WarningCollector
) -> Dict[str, Any]:
    """REM = DEM - surface on cells valid in both; negative values are preserved and counted."""
    n_valid = n_neg = n_nodata_surface = 0
    lo, hi = np.inf, -np.inf
    mean, m2 = 0.0, 0.0
    with rasterio.open(dem_path) as dem, rasterio.open(surface_path) as sf:
        out_nodata = dem.nodata
        tiled = dem.height >= TILE and dem.width >= TILE
        with create_raster(
            out_path, height=dem.height, width=dem.width, dtype="float32", transform=dem.transform, crs=dem.crs,
            nodata=out_nodata, compress="deflate", tiled=tiled, creation_options=_creation_options(),
            tags={"PRODUCT": "Relative Elevation Model: DEM minus IDW drainage surface (per basin)",
                  "METHOD": REM_METHOD_ID, "FORMULA": "REM = DEM_reference - Z_drainage_IDW_basin"},
            description="rem",
        ) as dst:
            for win in row_stripes(dem.height, dem.width, 48, align=TILE if tiled else 1):
                z = dem.read(1, window=win)
                s = sf.read(1, window=win)
                zv, sv = valid_mask(z, dem.nodata), valid_mask(s, sf.nodata)
                n_nodata_surface += int((zv & ~sv).sum())
                valid = zv & sv
                rem = np.full(z.shape, out_nodata, dtype=np.float32)
                d = z[valid].astype(np.float64) - s[valid].astype(np.float64)
                if not np.isfinite(d).all():
                    raise ProcessingError("The REM contains non-finite values; aborting.")
                rem[valid] = d.astype(np.float32)
                dst.write(rem, 1, window=win)
                if d.size:
                    n_neg += int((d < 0).sum())
                    lo, hi = min(lo, float(d.min())), max(hi, float(d.max()))
                    m_b = float(d.mean())
                    tot = n_valid + d.size
                    delta = m_b - mean
                    mean += delta * d.size / tot
                    m2 += float(((d - m_b) ** 2).sum()) + delta * delta * n_valid * d.size / tot
                    n_valid = tot
    if n_valid == 0:
        raise ProcessingError("The REM has no valid cells (every cell is nodata in the DEM, basins or surface).")
    ensure_same_grid(grid, out_path, "REM", warnings)
    # second pass: percentiles from a fine histogram of the stored (float32) REM
    counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    bin_width = (hi - lo) / HISTOGRAM_BINS if hi > lo else 0.0
    pct = {k: float(lo) for k in ("p05", "p25", "median", "p75", "p95")}
    if hi > lo:
        with rasterio.open(out_path) as r:
            for win in row_stripes(r.height, r.width, 16):
                a = r.read(1, window=win)
                v = a[valid_mask(a, r.nodata)].astype(np.float64)
                counts += np.histogram(v, bins=HISTOGRAM_BINS, range=(lo, hi))[0]
        pct = {k: _histogram_quantile(counts, lo, bin_width, q)
               for k, q in (("p05", 0.05), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p95", 0.95))}
    if n_nodata_surface:
        log.info("%d valid DEM cell(s) have no surface value (basins without samples) and are NoData in the REM.",
                 n_nodata_surface)
    if n_neg:
        log.info("REM: %d cell(s) (%.2f%%) are negative (DEM below the interpolated drainage surface); preserved.",
                 n_neg, 100 * n_neg / n_valid)
    stats = {
        "method": REM_METHOD_ID,
        "formula": "REM = DEM_reference - Z_drainage_IDW_basin",
        "reference_dem": "unconditioned DEM on the master grid (dem_processing_crs.tif)",
        "units": "vertical units of the DEM",
        "valid_cells": n_valid,
        "cells_nodata_because_no_surface": n_nodata_surface,
        "negative_cells": n_neg,
        "negative_fraction": n_neg / n_valid,
        "min": float(lo), "max": float(hi), "mean": float(mean), "std": float(np.sqrt(m2 / n_valid)),
        **pct,
        "percentile_method": f"histogram with {HISTOGRAM_BINS} bins (bin width {bin_width:.6g})",
        "dtype": "float32",
        "nodata": out_nodata,
        "note": "computed from the stored float32 surface; not thresholded, clipped or classified",
    }
    log.info("REM: min %.3f, median %.3f, max %.3f over %d cells.", stats["min"], stats["median"], stats["max"], n_valid)
    return stats
