"""Block-wise processing must give exactly the same results as whole-array processing."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from conftest import CRS, NODATA, RES, X0, Y0, config_dict, write_dem, write_gpkg
from floodplain_delineation import rasters
from floodplain_delineation.config import parse_config
from floodplain_delineation.delineation import apply_hand_threshold, build_final_mask, clip_to_pilot
from floodplain_delineation.exceptions import ProcessingError
from floodplain_delineation.inputs import prepare_spatial_domain, read_dem_info
from floodplain_delineation.logging_utils import WarningCollector
from floodplain_delineation.preprocessing import compare_conditioned, extract_processing_dem
from floodplain_delineation.rasters import normalize_geotiff, row_stripes
from floodplain_delineation.spatial import cell_geometry, to_pyproj


@pytest.fixture
def tiny_blocks(monkeypatch):
    """Force many stripes even on small test rasters."""
    monkeypatch.setattr(rasters, "BLOCK_BYTES", 2000)


def test_row_stripes_cover_all_rows_without_overlap(tiny_blocks):
    wins = list(row_stripes(103, 40, 8))
    assert len(wins) > 5
    assert sum(w.height for w in wins) == 103
    assert [w.row_off for w in wins] == np.cumsum([0] + [w.height for w in wins[:-1]]).tolist()
    assert all(w.width == 40 for w in wins)


def test_row_stripes_alignment_to_tile_height(tiny_blocks):
    wins = list(row_stripes(2000, 40, 8, align=512))
    assert all(int(w.height) % 512 == 0 for w in wins[:-1])


# ---------------------------------------------------------------- normalization ----------------
def test_normalize_geotiff_preserves_everything(tmp_path, tiny_blocks):
    arr = np.random.default_rng(0).normal(100, 20, (57, 43)).astype("float32")
    arr[3:6, 4:9] = NODATA
    src = write_dem(tmp_path / "raw.tif", arr)
    with rasterio.open(src, "r+") as d:
        d.update_tags(FOO="bar")
    dst = tmp_path / "norm.tif"
    normalize_geotiff(src, dst)
    with rasterio.open(src) as a, rasterio.open(dst) as b:
        assert np.array_equal(a.read(1), b.read(1))
        assert (a.crs, a.transform, a.nodata, a.dtypes, a.shape) == (b.crs, b.transform, b.nodata, b.dtypes, b.shape)
        assert b.compression is None and b.tags()["FOO"] == "bar"
    assert not (tmp_path / "norm.tif.partial").exists()


def test_normalize_geotiff_failure_leaves_no_partial_file(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_text("not a tiff")
    with pytest.raises(ProcessingError, match="Cannot rewrite"):
        normalize_geotiff(bad, tmp_path / "out.tif")
    assert not (tmp_path / "out.tif").exists() and not (tmp_path / "out.tif.partial").exists()


def test_run_raster_delivers_normalized_output_and_removes_raw(tmp_path, monkeypatch):
    from pathlib import Path
    from floodplain_delineation.wbt import WhiteboxRunner

    monkeypatch.setenv("FAKE_WBT_MODE", "copy")
    src = write_dem(tmp_path / "in.tif", np.arange(30, dtype="float32").reshape(5, 6))
    work = tmp_path / "work"; work.mkdir()
    out = tmp_path / "final.tif"
    WhiteboxRunner(Path(__file__).parent / "fake_wbt.py").run_raster(
        "BreachDepressions", [f"--dem={src}"], work_dir=work, output=out)
    assert out.is_file() and not list(work.glob("*"))
    with rasterio.open(src) as a, rasterio.open(out) as b:
        assert np.array_equal(a.read(1), b.read(1))


def test_run_raster_with_garbage_output_fails_cleanly(tmp_path, monkeypatch):
    from pathlib import Path
    from floodplain_delineation.wbt import WhiteboxRunner

    monkeypatch.setenv("FAKE_WBT_MODE", "garbage")
    work = tmp_path / "work"; work.mkdir()
    with pytest.raises(ProcessingError, match="Cannot rewrite"):
        WhiteboxRunner(Path(__file__).parent / "fake_wbt.py").run_raster(
            "BreachDepressions", ["--dem=a.tif"], work_dir=work, output=tmp_path / "o.tif")
    assert not (tmp_path / "o.tif").exists() and not list(work.glob("*"))


# ---------------------------------------------------------------- extraction ------------------
def _cfg(tmp_path, dem, gpkg, **over):
    return parse_config(config_dict(tmp_path, dem, gpkg, **over))


@pytest.mark.parametrize("dtype, nodata", [("float32", NODATA), ("int16", -32768), ("float32", float("nan"))])
def test_extraction_is_stripe_independent(tmp_path, tiny_blocks, dtype, nodata):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 500, (90, 70)).astype(dtype)
    arr[10:14, 20:30] = nodata
    dem = write_dem(tmp_path / "dem.tif", arr, nodata=nodata)
    gpkg = write_gpkg(tmp_path / "p.gpkg", [box(X0 + 5 * RES, Y0 - 80 * RES, X0 + 65 * RES, Y0 - 5 * RES)])
    cfg = _cfg(tmp_path, dem, gpkg, **{"processing.buffer_distance": 0})
    w = WarningCollector()
    info = read_dem_info(dem, w)
    domain = prepare_spatial_domain(info, cfg, w)
    out = tmp_path / "ext.tif"
    res = extract_processing_dem(info, domain, out, w)
    with rasterio.open(dem) as s, rasterio.open(out) as o:
        ref = s.read(1, window=domain.window)
        got = o.read(1)
        valid = ~np.isnan(ref) if np.isnan(nodata) else ref != nodata
        assert o.dtypes[0] == ("float32") and np.array_equal(got[valid], ref[valid].astype("float32"))
        assert (got[~valid] == o.nodata).all()
    assert res["valid_cells"] == int(valid.sum()) and res["nodata_cells"] == int((~valid).sum())
    from rasterio.features import geometry_mask
    inside = ~geometry_mask([domain.pilot], out_shape=ref.shape, transform=domain.window_transform)
    assert res["pilot_cells"] == int(inside.sum()) and res["pilot_valid_cells"] == int((inside & valid).sum())


# ---------------------------------------------------------------- conditioning diagnostics ----
def test_compare_conditioned_depth_statistics(tmp_path, tiny_blocks):
    orig = np.full((60, 50), 100.0, dtype="float32")
    cond = orig.copy()
    cond[5, 5] -= 0.5      # lowered by 0.5
    cond[6, 6] -= 3.0      # 3
    cond[7:9, 7] -= 30.0   # two cells lowered by 30
    cond[40, 40] += 2.0    # raised
    a = write_dem(tmp_path / "a.tif", orig); b = write_dem(tmp_path / "b.tif", cond)
    d = compare_conditioned(a, b, "BreachDepressions", [])
    assert d["cells_modified"] == 5 and d["cells_lowered"] == 4
    assert d["max_elevation_decrease"] == pytest.approx(30.0) and d["max_elevation_increase"] == pytest.approx(2.0)
    assert d["cells_lowered_more_than"] == {"1": 3, "5": 2, "10": 2, "25": 2, "50": 0}
    p = d["decrease_percentiles_approx"]
    assert 0.5 <= p["p50"] <= 3.5 and 30 <= p["p99"] <= 34       # upper bin edges (<= ~12 % above the value)


def test_compare_conditioned_detects_nodata_footprint_change(tmp_path):
    orig = np.full((20, 20), 10.0, dtype="float32"); cond = orig.copy(); cond[2, 2] = NODATA
    a = write_dem(tmp_path / "a.tif", orig); b = write_dem(tmp_path / "b.tif", cond)
    with pytest.raises(ProcessingError, match="nodata footprint"):
        compare_conditioned(a, b, "FillDepressionsWangAndLiu", ["--fix_flats"])


# ---------------------------------------------------------------- final mask ------------------
def test_final_mask_blockwise_equals_in_memory_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(rasters, "BLOCK_BYTES", 40_000)       # forces 512-row aligned stripes on a tiled output
    rng = np.random.default_rng(3)
    h, w = 1300, 1100
    hand = rng.uniform(0, 20, (h, w)).astype("float32")
    hand[100:200, 300:500] = NODATA
    hp = write_dem(tmp_path / "hand.tif", hand)
    pilot = Point(X0 + 550 * RES, Y0 - 650 * RES).buffer(480 * RES)          # round pilot, far from a bounding box
    cell = cell_geometry(to_pyproj(rasterio.crs.CRS.from_string(CRS)), from_origin(X0, Y0, RES, RES), (X0, Y0 - h * RES, X0 + w * RES, Y0))
    out = tmp_path / "final.tif"
    stats = build_final_mask(hp, pilot, 7.5, out, "meters", cell)
    mask = apply_hand_threshold(hand, hand != NODATA, 7.5)
    ref, ref_t = clip_to_pilot(mask, pilot, from_origin(X0, Y0, RES, RES))
    with rasterio.open(out) as o:
        assert o.block_shapes[0] == (512, 512) and o.compression.name == "deflate" and o.transform == ref_t
        assert np.array_equal(o.read(1), ref)
    assert stats["floodplain_cells"] == int((ref == 1).sum()) and stats["non_floodplain_cells"] == int((ref == 0).sum())
    assert stats["nodata_cells_in_extent"] == int((ref == 255).sum())
