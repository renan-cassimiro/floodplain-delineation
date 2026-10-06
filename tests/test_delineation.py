import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from conftest import CRS, RES, X0, Y0, write_dem
from floodplain_delineation.delineation import MASK_NODATA, apply_hand_threshold, build_final_mask, clip_to_pilot
from floodplain_delineation.exceptions import ProcessingError
from floodplain_delineation.spatial import cell_geometry, to_pyproj


def test_hand_threshold_spec_example():
    hand = np.array([[0, 2, 5], [7, 9, 3]], dtype="float32")
    mask = apply_hand_threshold(hand, np.ones(hand.shape, bool), 5)
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert mask.dtype == np.uint8


def test_threshold_is_inclusive_and_includes_negative_hand():
    hand = np.array([[-0.5, 4.999, 5.0, 5.001]], dtype="float64")
    assert apply_hand_threshold(hand, np.ones(hand.shape, bool), 5.0).tolist() == [[1, 1, 1, 0]]


def test_float32_hand_equal_to_threshold_counts_as_floodplain():
    hand = np.array([[np.float32(5.1)]])
    assert apply_hand_threshold(hand, np.ones((1, 1), bool), 5.1)[0, 0] == 1


def test_nodata_is_preserved_never_reclassified():
    hand = np.array([[0, -9999, 5], [7, np.nan, 3]], dtype="float32")
    valid = np.array([[True, False, True], [True, False, True]])
    mask = apply_hand_threshold(hand, valid, 5)
    assert mask.tolist() == [[1, MASK_NODATA, 1], [0, MASK_NODATA, 1]]


def test_clip_to_pilot_sets_outside_to_nodata_and_stays_grid_aligned():
    mask = np.ones((20, 20), dtype="uint8")
    tr = from_origin(X0, Y0, RES, RES)
    tri = box(X0 + 5.2 * RES, Y0 - 15.4 * RES, X0 + 15.1 * RES, Y0 - 5.3 * RES)
    # use a triangle so that some cells in the bounding window are outside the polygon
    from shapely.geometry import Polygon

    poly = Polygon([(X0 + 5 * RES, Y0 - 5 * RES), (X0 + 15 * RES, Y0 - 5 * RES), (X0 + 5 * RES, Y0 - 15 * RES)])
    out, t = clip_to_pilot(mask, poly, tr)
    assert out.shape[0] < 20 and (out == MASK_NODATA).any() and (out == 1).any()
    assert t.a == RES and t.e == -RES
    assert ((t.c - X0) / RES) == pytest.approx(round((t.c - X0) / RES))   # pixel alignment preserved
    assert ((Y0 - t.f) / RES) == pytest.approx(round((Y0 - t.f) / RES))


def test_pilot_smaller_than_a_cell_is_an_error():
    # cell centres sit at (n + 0.5) * RES; this polygon lies strictly between centres
    tiny = box(X0 + 5.1 * RES, Y0 - 5.4 * RES, X0 + 5.3 * RES, Y0 - 5.1 * RES)
    with pytest.raises(ProcessingError, match="smaller than one cell"):
        clip_to_pilot(np.ones((20, 20), "uint8"), tiny, from_origin(X0, Y0, RES, RES))


def _hand_raster(tmp_path, arr, nodata=-9999.0):
    return write_dem(tmp_path / "hand.tif", arr.astype("float64"), nodata=nodata)


def test_final_raster_metadata(tmp_path):
    hand = np.tile(np.arange(40, dtype="float64"), (30, 1)) / 4.0   # 0 .. 9.75
    hand[:3, :3] = -9999.0
    hp = _hand_raster(tmp_path, hand)
    pilot = box(X0 + 4 * RES, Y0 - 25 * RES, X0 + 36 * RES, Y0 - 4 * RES)
    cell = cell_geometry(to_pyproj(rasterio.crs.CRS.from_string(CRS)), from_origin(X0, Y0, RES, RES), (X0, Y0 - 30 * RES, X0 + 40 * RES, Y0))
    out = tmp_path / "potential_floodplain.tif"
    stats = build_final_mask(hp, pilot, 5.0, out, "meters", cell)
    with rasterio.open(out) as o, rasterio.open(hp) as h:
        assert o.dtypes[0] == "uint8" and o.nodata == 255
        assert o.crs == h.crs and o.res == h.res
        assert ((o.transform.c - h.transform.c) / RES) == pytest.approx(round((o.transform.c - h.transform.c) / RES))
        data = o.read(1)
        tags = o.tags()
    assert set(np.unique(data)) <= {0, 1, 255}
    assert "Preliminary" in tags["TITLE"] and "Not observed inundation" in tags["DESCRIPTION"]
    assert tags["HAND_THRESHOLD"] == "5.0" and tags["HAND_THRESHOLD_UNITS"] == "meters"
    assert stats["floodplain_cells"] == int((data == 1).sum())
    assert stats["floodplain_area_km2"] == pytest.approx(stats["floodplain_cells"] * 900 / 1e6)


def test_all_nodata_hand_inside_pilot_is_an_error(tmp_path):
    hp = _hand_raster(tmp_path, np.full((30, 30), -9999.0))
    cell = cell_geometry(to_pyproj(rasterio.crs.CRS.from_string(CRS)), from_origin(X0, Y0, RES, RES), (X0, Y0 - 900, X0 + 900, Y0))
    with pytest.raises(ProcessingError, match="entirely nodata"):
        build_final_mask(hp, box(X0 + 100, Y0 - 800, X0 + 800, Y0 - 100), 5, tmp_path / "o.tif", "meters", cell)
