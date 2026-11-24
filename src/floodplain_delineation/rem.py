"""Relative Elevation Model (REM) by basin-wise Euclidean IDW.

The drainage elevations are sampled from the conditioned DEM. Each basin is
interpolated independently; no samples from another basin enter an estimate.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio

from .exceptions import ProcessingError
from .logging_utils import get_logger
from .rasters import create_raster, ensure_same_grid, grid_of, row_stripes, valid_mask
from .wbt import WhiteboxRunner

log = get_logger("rem")
NODATA = -9999.0


def _sample_points(dem_path: Path, drainage_path: Path, basins_path: Path, interval: int):
    """Collect systematic drainage samples grouped by basin ID."""
    points = defaultdict(list)
    seen = defaultdict(int)
    with rasterio.open(dem_path) as dem, rasterio.open(drainage_path) as drain, rasterio.open(basins_path) as basins:
        for win in row_stripes(dem.height, dem.width, 20):
            z = dem.read(1, window=win)
            d = drain.read(1, window=win) == 1
            b = basins.read(1, window=win)
            valid = valid_mask(z, dem.nodata) & d & (b > 0)
            r0, c0 = int(win.row_off), int(win.col_off)
            rows, cols = np.nonzero(valid)
            if not len(rows):
                continue
            basin_values = b[rows, cols]
            for basin_id in np.unique(basin_values):
                idx = np.flatnonzero(basin_values == basin_id)
                # Deterministic systematic sampling in raster scan order, separately per basin.
                chosen = idx[(seen[int(basin_id)] + np.arange(len(idx))) % interval == 0]
                seen[int(basin_id)] += len(idx)
                if not len(chosen):
                    continue
                rr, cc = rows[chosen], cols[chosen]
                xs, ys = rasterio.transform.xy(dem.transform, rr + r0, cc + c0, offset="center")
                points[int(basin_id)].extend(zip(xs, ys, z[rr, cc].astype(float)))
    return points


def _idw_block(xy: np.ndarray, samples: np.ndarray, power: float, max_bytes: int = 48 * 1024 * 1024) -> np.ndarray:
    """Exact all-sample IDW, chunked over target cells to bound temporary memory."""
    out = np.empty(len(xy), dtype=np.float32)
    sx, sy, sz = samples[:, 0], samples[:, 1], samples[:, 2]
    n_samples = len(samples)
    chunk = max(1, min(len(xy), max_bytes // max(32 * n_samples, 32)))
    for start in range(0, len(xy), chunk):
        q = xy[start:start + chunk]
        dx = q[:, 0, None] - sx[None, :]
        dy = q[:, 1, None] - sy[None, :]
        d2 = dx * dx + dy * dy
        exact = d2 == 0
        has_exact = exact.any(axis=1)
        vals = np.empty(len(q), dtype=np.float64)
        if has_exact.any():
            ix = exact.argmax(axis=1)
            vals[has_exact] = sz[ix[has_exact]]
        other = ~has_exact
        if other.any():
            weights = np.power(d2[other], -power / 2.0)
            vals[other] = (weights @ sz) / weights.sum(axis=1)
        out[start:start + len(q)] = vals.astype(np.float32)
    return out


def compute_rem(
    runner: WhiteboxRunner, dem_path: Path, drainage_path: Path, flow_direction_path: Path,
    basins_path: Path, surface_path: Path, rem_path: Path, work_dir: Path,
    sampling_interval_pixels: int, idw_power: float, warnings,
) -> Dict[str, Any]:
    """Create basin IDs with WhiteboxTools, then interpolate REM by basin."""
    runner.run_raster(
        "Basins", [f"--d8_pntr={flow_direction_path}"], work_dir=work_dir, output=basins_path,
    )
    grid = grid_of(dem_path)
    ensure_same_grid(grid, basins_path, "REM basin IDs", warnings)
    ensure_same_grid(grid, drainage_path, "REM drainage network", warnings)
    with rasterio.open(basins_path) as basin_src:
        '''if not np.issubdtype(np.dtype(basin_src.dtypes[0]), np.integer):
            raise ProcessingError(f"REM basin raster must have an integer data type, got {basin_src.dtypes[0]}.")'''
        basin_ids = set()
        for win in row_stripes(basin_src.height, basin_src.width, 8):
            values = basin_src.read(1, window=win)
            basin_ids.update(int(v) for v in np.unique(values) if int(v) > 0)
    if not basin_ids:
        raise ProcessingError("WhiteboxTools Basins did not produce any positive basin IDs in the processing extent.")
    log.info("WhiteboxTools Basins produced %d positive basin ID(s).", len(basin_ids))
    points = _sample_points(dem_path, drainage_path, basins_path, sampling_interval_pixels)
    if not points:
        raise ProcessingError("REM: no valid drainage-elevation samples were found. Check the drainage threshold and sampling interval.")
    missing_samples = basin_ids.difference(points)
    if missing_samples:
        warnings.add(f"REM: {len(missing_samples)} basin(s) contain no sampled drainage elevations and will remain NoData.")
    log.info("REM: %d basin(s), %d with samples, %d sampled drainage cells.", len(basin_ids), len(points), sum(map(len, points.values())))

    with rasterio.open(dem_path) as dem, rasterio.open(basins_path) as basins:
        profile = dict(height=dem.height, width=dem.width, dtype="float32", transform=dem.transform,
                       crs=dem.crs, nodata=NODATA, compress="deflate", tiled=True)
        with create_raster(surface_path, **profile, tags={"METHOD": "basin-wise Euclidean IDW", "POWER": idw_power}) as surf, \
             create_raster(rem_path, **profile, tags={"FORMULA": "DEM - basin-wise IDW drainage elevation"}) as rem:
            for win in row_stripes(dem.height, dem.width, 32):
                z = dem.read(1, window=win)
                b = basins.read(1, window=win)
                valid = valid_mask(z, dem.nodata) & (b > 0)
                s_out = np.full(z.shape, NODATA, dtype=np.float32)
                r_out = np.full(z.shape, NODATA, dtype=np.float32)
                for basin_id in np.unique(b[valid]):
                    samples = np.asarray(points.get(int(basin_id), []), dtype=np.float64)
                    if samples.size == 0:
                        continue
                    mask = valid & (b == basin_id)
                    rows, cols = np.nonzero(mask)
                    xs, ys = rasterio.transform.xy(dem.transform, rows + int(win.row_off), cols + int(win.col_off), offset="center")
                    xy = np.column_stack((xs, ys)).astype(np.float64)
                    interp = _idw_block(xy, samples, idw_power)
                    s_out[rows, cols] = interp
                    r_out[rows, cols] = z[rows, cols].astype(np.float32) - interp
                surf.write(s_out, 1, window=win)
                rem.write(r_out, 1, window=win)
    return {
        "basins": len(basin_ids),
        "basins_with_samples": len(points),
        "samples": int(sum(len(v) for v in points.values())),
        "sampling_interval_pixels": sampling_interval_pixels,
        "idw_power": idw_power,
        "distance": "Euclidean in processing CRS map units",
        "elevation_source": "conditioned DEM",
        "interpolation": "exact all-sample IDW independently within each WhiteboxTools Basins region",
    }
