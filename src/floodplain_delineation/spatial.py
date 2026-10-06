"""CRS, cell-size and cell-area helpers.

All areas/lengths reported by the workflow go through these helpers so that
projected CRSs with non-metre units and geographic CRSs are handled explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import pyproj
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine

from .exceptions import InputError


def to_pyproj(crs: RioCRS) -> pyproj.CRS:
    try:
        return pyproj.CRS.from_wkt(crs.to_wkt())
    except Exception as exc:  # pragma: no cover - defensive
        raise InputError(f"Unsupported or unreadable coordinate reference system: {exc}") from exc


def check_supported_crs(crs: Optional[RioCRS], what: str) -> pyproj.CRS:
    """Require a defined, projected or geographic CRS."""
    if crs is None:
        raise InputError(
            f"{what} has no coordinate reference system. Assign one before running "
            "(e.g. `gdal_edit.py -a_srs EPSG:xxxx file.tif`)."
        )
    pcrs = to_pyproj(crs)
    if not (pcrs.is_projected or pcrs.is_geographic):
        raise InputError(
            f"{what} uses an unsupported CRS type ({pcrs.type_name}). "
            "Only projected and geographic CRSs are supported."
        )
    return pcrs


def linear_units_factor(pcrs: pyproj.CRS) -> float:
    """Metres per CRS horizontal unit (projected CRSs only)."""
    return float(pcrs.axis_info[0].unit_conversion_factor)


@dataclass(frozen=True)
class CellGeometry:
    """Approximate cell dimensions in metres at a reference location."""

    dx_m: float
    dy_m: float
    area_m2: float
    geographic: bool
    reference_latitude: Optional[float]
    area_variation_fraction: Optional[float]  # (max-min)/mean across the extent; geographic only

    @property
    def area_km2(self) -> float:
        return self.area_m2 / 1e6


def cell_geometry(pcrs: pyproj.CRS, transform: Affine, bounds: Tuple[float, float, float, float]) -> CellGeometry:
    """Cell size/area for a raster grid.

    Projected CRS: exact (constant) from the affine transform and axis unit.
    Geographic CRS: geodesic dimensions at the extent's central latitude, plus the
    relative variation of cell area between the extent's southern and northern edges.
    """
    left, bottom, right, top = bounds
    if pcrs.is_projected:
        f = linear_units_factor(pcrs)
        dx, dy = abs(transform.a) * f, abs(transform.e) * f
        return CellGeometry(dx, dy, dx * dy, False, None, None)

    geod = pcrs.get_geod()
    lon0 = (left + right) / 2.0

    def dims(lat: float) -> Tuple[float, float]:
        lat = max(min(lat, 89.999), -89.999)
        dx = geod.inv(lon0, lat, lon0 + abs(transform.a), lat)[2]
        dy = geod.inv(lon0, lat, lon0, lat + abs(transform.e))[2]
        return dx, dy

    lat_c = (bottom + top) / 2.0
    dx, dy = dims(lat_c)
    areas = [d[0] * d[1] for d in (dims(bottom), dims(top))]
    mean = sum(areas) / 2.0
    variation = (max(areas) - min(areas)) / mean if mean > 0 else None
    return CellGeometry(dx, dy, dx * dy, True, lat_c, variation)


def buffer_distance_in_crs_units(
    distance: float, units: str, pcrs: pyproj.CRS, transform: Affine
) -> Optional[float]:
    """Convert a buffer distance to DEM CRS units when that is a direct operation.

    Returns ``None`` when a geographic CRS needs a metric buffer (handled by the caller
    in a suitable projected CRS).
    """
    if units == "cells":
        return distance * max(abs(transform.a), abs(transform.e))
    if pcrs.is_projected:
        return distance / linear_units_factor(pcrs)
    return None


def degrees_note() -> str:  # pragma: no cover - trivial
    return "geographic"


def sqrt2_length(a: float, b: float) -> float:
    return math.hypot(a, b)
