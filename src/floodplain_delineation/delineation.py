"""Stages 6-7: HAND threshold -> binary mask, clipping to the pilot area, final raster.

Product: *preliminary potentially inundatable floodplain map* (topographic approximation).
Classes: 1 = HAND <= threshold, 0 = HAND > threshold, 255 = nodata (never reclassified).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry.base import BaseGeometry

from .exceptions import ProcessingError
from .logging_utils import get_logger
from .rasters import read_band, write_raster
from .spatial import CellGeometry

log = get_logger("delineation")

MASK_NODATA = 255
PRODUCT_TITLE = "Preliminary potentially inundatable floodplain map (HAND threshold)"
PRODUCT_DESCRIPTION = (
    "Topographic approximation: 1 where HAND <= threshold, 0 where HAND > threshold, 255 = nodata. "
    "Not observed inundation, not a hydraulic simulation, not a validated floodplain boundary."
)


def apply_hand_threshold(hand: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    """Binary mask (uint8) from HAND. Invalid cells become ``MASK_NODATA`` (never 0 or 1).

    The comparison is inclusive (``HAND <= threshold``) and done in the HAND array's own
    floating-point precision, so a stored value equal to the threshold is floodplain.
    """
    if hand.shape != valid.shape:
        raise ProcessingError("HAND array and validity mask must have the same shape.")
    if not np.issubdtype(hand.dtype, np.floating):
        hand = hand.astype(np.float64)
    out = np.full(hand.shape, MASK_NODATA, dtype=np.uint8)
    thr = hand.dtype.type(threshold)
    with np.errstate(invalid="ignore"):
        out[valid] = (hand[valid] <= thr).astype(np.uint8)
    return out


def pilot_window(pilot: BaseGeometry, transform, shape: Tuple[int, int]) -> Window:
    """Smallest integer window (on the processing grid) covering the pilot geometry."""
    w = from_bounds(*pilot.bounds, transform=transform)
    eps = 1e-6
    c0 = max(0, math.floor(w.col_off + eps))
    r0 = max(0, math.floor(w.row_off + eps))
    c1 = min(shape[1], math.ceil(w.col_off + w.width - eps))
    r1 = min(shape[0], math.ceil(w.row_off + w.height - eps))
    if c1 <= c0 or r1 <= r0:
        raise ProcessingError("Pilot area lies outside the processing grid.")
    return Window(c0, r0, c1 - c0, r1 - r0)


def clip_to_pilot(
    mask: np.ndarray, pilot: BaseGeometry, transform, nodata: int = MASK_NODATA
) -> Tuple[np.ndarray, Any]:
    """Crop to the pilot bounding window (grid-aligned) and set cells outside it to nodata.

    A cell belongs to the pilot area when its centre lies inside the polygon.
    Returns ``(clipped_array, clipped_transform)``.
    """
    win = pilot_window(pilot, transform, mask.shape)
    sub_t = window_transform(win, transform)
    r0, c0 = int(win.row_off), int(win.col_off)
    sub = mask[r0 : r0 + int(win.height), c0 : c0 + int(win.width)].copy()
    inside = ~geometry_mask([pilot], out_shape=sub.shape, transform=sub_t, all_touched=False)
    if not inside.any():
        raise ProcessingError("The pilot area is smaller than one cell; no cell centre falls inside it.")
    sub[~inside] = nodata
    return sub, sub_t


def build_final_mask(
    hand_path: Path, pilot: BaseGeometry, threshold: float, out_path: Path,
    vertical_units: str, cell: CellGeometry,
) -> Dict[str, Any]:
    """Create ``potential_floodplain.tif`` from the HAND raster."""
    hand, valid, grid, _ = read_band(hand_path)
    mask = apply_hand_threshold(hand, valid, threshold)
    clipped, tr = clip_to_pilot(mask, pilot, grid.transform)

    n_fp = int((clipped == 1).sum())
    n_non = int((clipped == 0).sum())
    if n_fp + n_non == 0:
        raise ProcessingError(
            "No valid HAND cells fall inside the pilot area; the floodplain mask would be entirely nodata."
        )
    write_raster(
        out_path, clipped, transform=tr, crs=grid.crs, nodata=MASK_NODATA,
        tags={
            "TITLE": PRODUCT_TITLE,
            "DESCRIPTION": PRODUCT_DESCRIPTION,
            "CLASS_VALUES": "1=potential floodplain (HAND<=threshold); 0=outside; 255=nodata",
            "HAND_THRESHOLD": threshold,
            "HAND_THRESHOLD_UNITS": vertical_units,
        },
        description="potential_floodplain",
    )
    area_factor = cell.area_km2
    log.info("Potential floodplain: %d cells (%.3f km2) of %d valid cells in the pilot area.",
             n_fp, n_fp * area_factor, n_fp + n_non)
    return {
        "path": str(out_path),
        "shape": [int(clipped.shape[0]), int(clipped.shape[1])],
        "dtype": "uint8",
        "nodata": MASK_NODATA,
        "floodplain_cells": n_fp,
        "non_floodplain_cells": n_non,
        "nodata_cells_in_extent": int((clipped == MASK_NODATA).sum()),
        "floodplain_fraction_of_valid": n_fp / (n_fp + n_non),
        "floodplain_area_km2": n_fp * area_factor,
        "area_note": "cell-count x nominal cell area" + (" (approximate: geographic CRS)" if cell.geographic else ""),
    }
