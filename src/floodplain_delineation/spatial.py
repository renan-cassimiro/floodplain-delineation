"""CRS, cell-size and cell-area helpers.

All areas/lengths reported by the workflow go through these helpers so that
projected CRSs with non-metre units and geographic CRSs are handled explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyproj
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine
from shapely.geometry import Polygon

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


# --------------------------------------------------------------------------- #
# Processing CRS (Stage 0)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProcessingCrs:
    """The projected CRS in which the whole pipeline runs, with the facts that go to the summary."""

    spec: str                  # as configured, e.g. "ESRI:102033"
    pcrs: pyproj.CRS
    crs: RioCRS
    name: str
    authority: Optional[str]   # e.g. "ESRI:102033" (never relabelled as EPSG)
    unit_name: str
    unit_factor: float         # metres per CRS unit

    def describe(self) -> Dict[str, Any]:
        return {
            "configured": self.spec,
            "name": self.name,
            "authority_code": self.authority,
            "type": self.pcrs.type_name,
            "horizontal_units": self.unit_name,
            "metres_per_unit": self.unit_factor,
            "proj_version": pyproj.proj_version_str,
        }


def resolve_processing_crs(spec: str) -> ProcessingCrs:
    """Resolve ``processing.crs`` with the installed PROJ; it must be projected with linear units.

    No substitution is ever made: an unknown identifier (e.g. ``ESRI:102033`` on a PROJ without the
    ESRI database) is an error, and ``EPSG:102033`` is a different (non-existent) identifier.
    """
    try:
        pcrs = pyproj.CRS.from_user_input(spec)
    except Exception as exc:
        raise InputError(
            f"processing.crs '{spec}' is not recognised by the installed PROJ ({pyproj.proj_version_str}): {exc}. "
            "No other CRS is substituted. Note that 'ESRI:102033' and 'EPSG:102033' are different identifiers."
        ) from exc
    if not pcrs.is_projected:
        raise InputError(
            f"processing.crs '{spec}' is a {pcrs.type_name}; a projected CRS with linear units is required "
            "because the REM/IDW distances are Euclidean distances in CRS units."
        )
    axis = pcrs.axis_info[0]
    factor = float(axis.unit_conversion_factor)
    if not (math.isfinite(factor) and factor > 0):
        raise InputError(f"processing.crs '{spec}' has no usable linear horizontal unit.")
    try:
        crs = RioCRS.from_wkt(pcrs.to_wkt())
    except Exception as exc:  # pragma: no cover - defensive
        raise InputError(f"processing.crs '{spec}' cannot be represented by GDAL: {exc}") from exc
    auth = pcrs.to_authority()
    return ProcessingCrs(
        spec=spec, pcrs=pcrs, crs=crs, name=pcrs.name,
        authority=f"{auth[0]}:{auth[1]}" if auth else None,
        unit_name=axis.unit_name, unit_factor=factor,
    )


def same_crs(a: pyproj.CRS, b: pyproj.CRS) -> bool:
    return bool(a == b or a.equals(b, ignore_axis_order=True))


def densified_footprint(
    bounds: Tuple[float, float, float, float], source: pyproj.CRS, target: pyproj.CRS, points_per_edge: int = 21
) -> Polygon:
    """Footprint of a rectangle in ``target`` CRS, from its edges densified with ``points_per_edge`` points.

    Transforming only the four corners underestimates the extent when the transformation is non-linear
    (meridian convergence, projection curvature); the densified boundary follows the real shape.
    """
    left, bottom, right, top = bounds
    t = np.linspace(0.0, 1.0, points_per_edge + 1, endpoint=False)
    xs = np.concatenate([left + (right - left) * t, np.full_like(t, right), right - (right - left) * t, np.full_like(t, left)])
    ys = np.concatenate([np.full_like(t, bottom), bottom + (top - bottom) * t, np.full_like(t, top), top - (top - bottom) * t])
    if same_crs(source, target):
        tx, ty = xs, ys
    else:
        tr = pyproj.Transformer.from_crs(source, target, always_xy=True)
        tx, ty = tr.transform(xs, ys)
    tx, ty = np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)
    if not (np.isfinite(tx).all() and np.isfinite(ty).all()):
        raise InputError(
            "The DEM extent cannot be transformed to the processing CRS (non-finite coordinates): the DEM lies "
            "partly outside the validity domain of that projection. Choose another processing.crs."
        )
    poly = Polygon(np.column_stack([tx, ty]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly
