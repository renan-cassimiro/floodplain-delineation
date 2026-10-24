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
from .rasters import create_raster, row_stripes, shift_rows, valid_mask
from .spatial import CellGeometry

log = get_logger("delineation")

MASK_NODATA = 255
TILE = 512   # tile size of the (large) final raster
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
    """Create ``potential_floodplain.tif`` from the HAND raster, block by block.

    Only the pilot's bounding window is read and written; a cell belongs to the pilot area when
    its centre lies inside the polygon. The output is tiled (512x512) and DEFLATE-compressed when
    the window is large enough for that to make sense.
    """
    with rasterio.open(hand_path) as src:
        win = pilot_window(pilot, src.transform, (src.height, src.width))
        sub_t = window_transform(win, src.transform)
        height, width = int(win.height), int(win.width)
        tiled = height >= TILE and width >= TILE
        pilot_miny, pilot_maxy = pilot.bounds[1], pilot.bounds[3]
        n_fp = n_non = n_nodata = n_inside = 0
        try:
            with create_raster(
                out_path, height=height, width=width, dtype="uint8", transform=sub_t, crs=src.crs,
                nodata=MASK_NODATA, compress="deflate", tiled=tiled,
                tags={
                    "TITLE": PRODUCT_TITLE,
                    "DESCRIPTION": PRODUCT_DESCRIPTION,
                    "CLASS_VALUES": "1=potential floodplain (HAND<=threshold); 0=outside; 255=nodata",
                    "HAND_THRESHOLD": threshold,
                    "HAND_THRESHOLD_UNITS": vertical_units,
                },
                description="potential_floodplain",
            ) as dst:
                for stripe in row_stripes(height, width, 24, align=TILE if tiled else 1):
                    r0, rows = int(stripe.row_off), int(stripe.height)
                    hand = src.read(1, window=Window(win.col_off, win.row_off + r0, width, rows))
                    mask = apply_hand_threshold(hand, valid_mask(hand, src.nodata), threshold)
                    t = shift_rows(sub_t, r0)
                    if t.f + t.e * rows <= pilot_maxy and t.f >= pilot_miny:
                        inside = ~geometry_mask([pilot], out_shape=mask.shape, transform=t, all_touched=False)
                    else:
                        inside = np.zeros(mask.shape, dtype=bool)
                    mask[~inside] = MASK_NODATA
                    n_inside += int(inside.sum())
                    n_fp += int((mask == 1).sum())
                    n_non += int((mask == 0).sum())
                    n_nodata += int((mask == MASK_NODATA).sum())
                    dst.write(mask, 1, window=stripe)
                if n_inside == 0:
                    raise ProcessingError("The pilot area is smaller than one cell; no cell centre falls inside it.")
                if n_fp + n_non == 0:
                    raise ProcessingError(
                        "No valid HAND cells fall inside the pilot area; the floodplain mask would be entirely nodata."
                    )
        except Exception:
            out_path.unlink(missing_ok=True)
            raise
    area_factor = cell.area_km2
    log.info("Potential floodplain: %d cells (%.3f km2) of %d valid cells in the pilot area.",
             n_fp, n_fp * area_factor, n_fp + n_non)
    return {
        "path": str(out_path),
        "shape": [height, width],
        "dtype": "uint8",
        "nodata": MASK_NODATA,
        "floodplain_cells": n_fp,
        "non_floodplain_cells": n_non,
        "nodata_cells_in_extent": n_nodata,
        "floodplain_fraction_of_valid": n_fp / (n_fp + n_non),
        "floodplain_area_km2": n_fp * area_factor,
        "area_note": "cell-count x nominal cell area" + (" (approximate: geographic CRS)" if cell.geographic else ""),
    }
