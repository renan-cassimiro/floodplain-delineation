import math

import numpy as np
import pytest
import pyproj
from rasterio.transform import from_origin

from conftest import RES, X0, Y0, write_dem
from floodplain_delineation.exceptions import ProcessingError
from floodplain_delineation.hydrology import drainage_diagnostics, resolve_drainage_threshold
from floodplain_delineation.logging_utils import WarningCollector
from floodplain_delineation.spatial import cell_geometry


@pytest.fixture
def utm_cell():
    return cell_geometry(pyproj.CRS.from_epsg(32720), from_origin(X0, Y0, RES, RES), (X0, Y0 - 3000, X0 + 3000, Y0))


def test_cells_threshold_is_used_as_is(utm_cell):
    t = resolve_drainage_threshold("cells", 250, utm_cell, WarningCollector())
    assert t["effective_cells"] == 250 and t["effective_area_km2"] == pytest.approx(250 * 900 / 1e6)


def test_area_threshold_converted_with_actual_cell_area_rounding_up(utm_cell):
    # 1 km2 / 900 m2 = 1111.1 cells -> need 1112 cells so contributing area >= 1 km2
    t = resolve_drainage_threshold("area_km2", 1.0, utm_cell, WarningCollector())
    assert t["effective_cells"] == 1112 and t["effective_area_km2"] >= 1.0


def test_area_threshold_exact_multiple_does_not_round_up(utm_cell):
    t = resolve_drainage_threshold("area_km2", 0.0009, utm_cell, WarningCollector())  # exactly 1 cell
    assert t["effective_cells"] == 1


def test_area_threshold_in_geographic_crs_warns_with_variation():
    cell = cell_geometry(pyproj.CRS.from_epsg(4326), from_origin(-60, 0, 0.0009, 0.0009), (-60, -1, -59, 0))
    w = WarningCollector()
    t = resolve_drainage_threshold("area_km2", 1.0, cell, w)
    assert t["effective_cells"] > 0
    assert any("geographic CRS" in m and "projected CRS is recommended" in m for m in w.messages)


def test_drainage_diagnostics_length_and_percentage(tmp_path, utm_cell):
    # 3 eastward steps (code 2) + 2 diagonal steps (code 4) on drainage cells
    ptr = np.zeros((4, 6), dtype="int16")
    drain = np.zeros((4, 6), dtype="uint8")
    drain[1, 0:3] = 1; ptr[1, 0:3] = 2
    drain[2, 3:5] = 1; ptr[2, 3:5] = 4
    drain[3, 5] = 255                       # nodata cell excluded from the percentage
    p = write_dem(tmp_path / "ptr.tif", ptr, nodata=-32768)
    d = drainage_diagnostics(drain, p, utm_cell)
    assert d["drainage_cells"] == 5 and d["valid_cells"] == 23
    assert d["percent_of_processing_area"] == pytest.approx(100 * 5 / 23)
    assert d["approx_length_km"] == pytest.approx((3 * 30 + 2 * 30 * math.sqrt(2)) / 1000)
