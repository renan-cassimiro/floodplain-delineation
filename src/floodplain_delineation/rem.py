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
    runner: WhiteboxRunner,
    dem_path: Path,
    drainage_path: Path,
    flow_direction_path: Path,
    basins_path: Path,
    surface_path: Path,
    rem_path: Path,
    work_dir: Path,
    sampling_interval_pixels: int,
    idw_power: float,
    warnings,
    idw_neighbors: int = 16,
    idw_max_distance: float = None,
) -> Dict[str, Any]:
    """
    Create basin IDs with WhiteboxTools, then interpolate REM
    using basin-wise K-nearest-neighbor IDW.

    Spatial indexes are built once per basin. Raster processing
    is performed in blocks to control memory consumption.
    """
    from scipy.spatial import cKDTree

    # --------------------------------------------------
    # 1. Validate interpolation parameters
    # --------------------------------------------------

    if idw_power <= 0:
        raise ProcessingError("REM: IDW power must be greater than zero.")

    if idw_neighbors < 1:
        raise ProcessingError("REM: IDW neighbors must be at least 1.")

    if idw_max_distance is not None and idw_max_distance <= 0:
        raise ProcessingError(
            "REM: maximum IDW distance must be greater than zero."
        )

    max_distance = (
        np.inf if idw_max_distance is None else idw_max_distance
    )

    # --------------------------------------------------
    # 2. Generate basin IDs
    # --------------------------------------------------

    runner.run_raster(
        "Basins",
        [f"--d8_pntr={flow_direction_path}"],
        work_dir=work_dir,
        output=basins_path,
    )

    grid = grid_of(dem_path)

    ensure_same_grid(grid, basins_path, "REM basin IDs", warnings)
    ensure_same_grid(grid, drainage_path, "REM drainage network", warnings)

    # --------------------------------------------------
    # 3. Sample drainage elevations
    # --------------------------------------------------

    points = _sample_points(
        dem_path,
        drainage_path,
        basins_path,
        sampling_interval_pixels,
    )

    if not points:
        raise ProcessingError(
            "REM: no valid drainage-elevation samples were found. "
            "Check the drainage threshold and sampling interval."
        )

    # --------------------------------------------------
    # 4. Build one spatial index per basin
    # --------------------------------------------------

    basin_trees = {}
    total_samples = 0

    for basin_id, samples in points.items():

        samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)

        # Remove invalid sample coordinates or elevations
        samples = samples[np.isfinite(samples).all(axis=1)]

        if len(samples) == 0:
            continue

        basin_trees[int(basin_id)] = (
            cKDTree(samples[:, :2]),
            samples[:, 2],
        )

        total_samples += len(samples)

    if not basin_trees:
        raise ProcessingError(
            "REM: no valid basin-specific spatial indexes could be built."
        )

    log.info(
        "REM: built %d basin spatial indexes with %d samples.",
        len(basin_trees),
        total_samples,
    )

    # --------------------------------------------------
    # 5. Open rasters and prepare outputs
    # --------------------------------------------------

    with rasterio.open(dem_path) as dem, \
         rasterio.open(basins_path) as basins:

        profile = dict(height=dem.height, width=dem.width, dtype="float32", transform=dem.transform,
            crs=dem.crs, nodata=NODATA, compress="deflate", tiled=True,
        )

        basin_ids = set()

        with create_raster(
            surface_path,
            **profile,
            tags={
                "METHOD": "Basin-wise KNN IDW",
                "POWER": str(idw_power),
                "NEIGHBORS": str(idw_neighbors),
                "MAX_DISTANCE": str(idw_max_distance),
            },
        ) as surf, create_raster(
            rem_path,
            **profile,
            tags={
                "FORMULA": "DEM - basin-wise KNN IDW drainage elevation",
            },
        ) as rem:

            # --------------------------------------------------
            # 6. Process raster in blocks
            # --------------------------------------------------

            for win in row_stripes(dem.height, dem.width, 32):

                z = dem.read(1, window=win)
                b = basins.read(1, window=win)

                valid = valid_mask(z, dem.nodata) & (b > 0)

                s_out = np.full(
                    z.shape, NODATA, dtype=np.float32
                )

                r_out = np.full(
                    z.shape, NODATA, dtype=np.float32
                )

                if valid.any():

                    # Collect basin IDs during the processing pass
                    basin_ids.update(
                        int(v) for v in np.unique(b[valid])
                    )

                    for basin_id in np.unique(b[valid]):

                        basin_id = int(basin_id)

                        if basin_id not in basin_trees:
                            continue

                        tree, elevations = basin_trees[basin_id]

                        # Identify valid pixels belonging to this basin
                        mask = valid & (b == basin_id)
                        rows, cols = np.nonzero(mask)

                        # Convert local pixel indices to global indices
                        global_rows = rows + int(win.row_off)
                        global_cols = cols + int(win.col_off)

                        # Calculate pixel-center coordinates directly
                        t = dem.transform

                        pixel_cols = global_cols + 0.5
                        pixel_rows = global_rows + 0.5

                        xs = (
                            t.a * pixel_cols
                            + t.b * pixel_rows
                            + t.c
                        )

                        ys = (
                            t.d * pixel_cols
                            + t.e * pixel_rows
                            + t.f
                        )

                        xy = np.column_stack((xs, ys))

                        # Query nearest drainage samples
                        k = min(idw_neighbors, len(elevations))

                        distances, indices = tree.query(
                            xy,
                            k=k,
                            distance_upper_bound=max_distance,
                            workers=-1,
                        )

                        # Normalize dimensions for k=1
                        if k == 1:
                            distances = distances[:, None]
                            indices = indices[:, None]

                        # Identify valid neighbors
                        neighbor_valid = (
                            np.isfinite(distances)
                            & (indices < len(elevations))
                        )

                        has_neighbors = neighbor_valid.any(axis=1)

                        if not has_neighbors.any():
                            continue

                        interp = np.full(
                            len(xy), np.nan, dtype=np.float64
                        )

                        # Exact sample locations
                        exact = neighbor_valid & (distances == 0)
                        has_exact = exact.any(axis=1)

                        if has_exact.any():

                            exact_rows = np.flatnonzero(has_exact)

                            exact_cols = exact[
                                exact_rows
                            ].argmax(axis=1)

                            interp[exact_rows] = elevations[
                                indices[exact_rows, exact_cols]
                            ]

                        # IDW for non-exact locations
                        interpolate = has_neighbors & ~has_exact

                        if interpolate.any():

                            d = distances[interpolate]
                            idx = indices[interpolate]
                            v = neighbor_valid[interpolate]

                            # Scale distances to improve numerical stability
                            min_distance = np.min(
                                np.where(v, d, np.inf),
                                axis=1,
                            )

                            weights = np.zeros_like(
                                d, dtype=np.float64
                            )

                            weights[v] = (
                                np.repeat(
                                    min_distance[:, None],
                                    k,
                                    axis=1,
                                )[v] / d[v]
                            ) ** idw_power

                            sample_z = np.where(
                                v, elevations[idx], 0.0
                            )

                            interp[interpolate] = (
                                np.sum(weights * sample_z, axis=1)
                                / np.sum(weights, axis=1)
                            )

                        # Write valid interpolated pixels
                        valid_interp = np.isfinite(interp)

                        target_rows = rows[valid_interp]
                        target_cols = cols[valid_interp]

                        s_out[target_rows, target_cols] = (
                            interp[valid_interp].astype(np.float32)
                        )

                        r_out[target_rows, target_cols] = (
                            z[target_rows, target_cols].astype(
                                np.float64
                            ) - interp[valid_interp]
                        ).astype(np.float32)

                # Write completed block
                surf.write(s_out, 1, window=win)
                rem.write(r_out, 1, window=win)

    # --------------------------------------------------
    # 7. Report missing basin samples
    # --------------------------------------------------

    missing_samples = basin_ids.difference(basin_trees)

    if missing_samples:
        warnings.add(
            f"REM: {len(missing_samples)} basin(s) contain no "
            "sampled drainage elevations and will remain NoData."
        )

    basins_with_samples = len(basin_ids.intersection(basin_trees))

    log.info(
        "REM: %d basin(s), %d with samples, %d sampled drainage cells.",
        len(basin_ids),
        basins_with_samples,
        total_samples,
    )

    return {
        "basins": len(basin_ids),
        "basins_with_samples": basins_with_samples,
        "samples": total_samples,
        "sampling_interval_pixels": sampling_interval_pixels,
        "idw_power": idw_power,
        "idw_neighbors": idw_neighbors,
        "idw_max_distance": idw_max_distance,
        "distance": "Euclidean in processing CRS map units",
        "elevation_source": "conditioned DEM",
        "interpolation": (
            "Basin-wise KNN IDW using prebuilt spatial indexes"
        ),
    }
