"""REM: IDW, drainage sampling, per-basin separation, Subbasins handling, REM computation (no WhiteboxTools needed)."""

import numpy as np
import pytest
import rasterio

from conftest import NODATA, RES, X0, Y0, write_dem
from floodplain_delineation import rasters, rem
from floodplain_delineation.exceptions import ProcessingError
from floodplain_delineation.logging_utils import WarningCollector
from floodplain_delineation.rasters import grid_of
from floodplain_delineation.rem import (
    SampleTable, compute_rem, delineate_basins, idw_estimate, interpolate_surface, sample_drainage,
    select_block_samples,
)

BN = -9999  # nodata of basin rasters written by these tests


def manual_idw(tx, ty, sx, sy, sz, p):
    d = np.hypot(tx - sx, ty - sy)
    if (d == 0).any():
        return float(sz[np.argmin(d)])
    w = 1.0 / d ** p
    return float((w * sz).sum() / w.sum())


# ------------------------------------------------------------------ IDW ----------------
SX = np.array([0.0, 100.0, 0.0, 100.0]); SY = np.array([0.0, 0.0, 100.0, 100.0]); SZ = np.array([10.0, 20.0, 30.0, 40.0])


def test_idw_four_samples_matches_manual_calculation():
    tx = np.array([50.0, 20.0, 80.0, 30.0]); ty = np.array([50.0, 10.0, 90.0, 70.0])
    got = idw_estimate(tx, ty, SX, SY, SZ, 2.0)
    want = [manual_idw(a, b, SX, SY, SZ, 2.0) for a, b in zip(tx, ty)]
    assert got == pytest.approx(want, rel=1e-12)
    assert got[0] == pytest.approx(25.0)             # centre: equal weights -> arithmetic mean


def test_idw_cell_on_a_sample_gets_exactly_its_elevation():
    got = idw_estimate(np.array([100.0, 0.0]), np.array([0.0, 100.0]), SX, SY, SZ, 2.0)
    assert got[0] == 20.0 and got[1] == 30.0         # exact, no 0/0, no inf


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.7])
def test_idw_power_changes_the_weights_as_expected(p):
    tx, ty = np.array([20.0]), np.array([10.0])
    assert idw_estimate(tx, ty, SX, SY, SZ, p)[0] == pytest.approx(manual_idw(20, 10, SX, SY, SZ, p), rel=1e-12)
    # a larger exponent weights the nearest sample (z=10 at the origin) more: the estimate moves toward 10
    assert idw_estimate(tx, ty, SX, SY, SZ, p + 1)[0] < idw_estimate(tx, ty, SX, SY, SZ, p)[0]


def test_idw_high_exponent_does_not_underflow():
    near = idw_estimate(np.array([90.0]), np.array([95.0]), SX, SY, SZ, 60.0)
    assert near[0] == pytest.approx(40.0, abs=1e-6)                            # the nearest sample (100,100) dominates
    far = idw_estimate(np.array([1e5]), np.array([1e5]), SX, SY, SZ, 60.0)     # w=d^-60 would underflow to 0 -> 0/0
    assert np.isfinite(far[0]) and 10.0 <= far[0] <= 40.0


def test_idw_uses_every_sample_and_is_independent_of_chunking(monkeypatch):
    rng = np.random.default_rng(0)
    sx, sy, sz = rng.uniform(0, 1e4, 300), rng.uniform(0, 1e4, 300), rng.uniform(0, 100, 300)
    tx, ty = rng.uniform(0, 1e4, 57), rng.uniform(0, 1e4, 57)
    full = idw_estimate(tx, ty, sx, sy, sz, 2.0)
    assert full == pytest.approx([manual_idw(a, b, sx, sy, sz, 2.0) for a, b in zip(tx, ty)], rel=1e-10)
    monkeypatch.setattr(rem, "IDW_PAIR_BUDGET", 700)                    # forces many chunks of a few targets
    assert idw_estimate(tx, ty, sx, sy, sz, 2.0) == pytest.approx(full, rel=1e-12, abs=0)   # only summation order differs
    # no truncation: dropping the farthest sample changes every estimate (it takes part in all of them)
    far = np.argmax(np.hypot(sx - tx.mean(), sy - ty.mean()))
    keep = np.arange(300) != far
    assert (idw_estimate(tx, ty, sx[keep], sy[keep], sz[keep], 2.0) != full).all()


# ------------------------------------------------------------ block selection ----------
def test_interval_one_keeps_every_candidate():
    rows, cols = np.nonzero(np.ones((5, 7), bool))
    basin = np.ones(rows.size, np.int64)
    sel = select_block_samples(rows, cols, basin, 1, 30.0, 30.0)
    assert sorted(sel.tolist()) == list(range(rows.size))


def test_selection_is_order_independent_nearest_to_centre_and_one_per_block_and_basin():
    rng = np.random.default_rng(3)
    rows, cols = rng.integers(0, 40, 400), rng.integers(0, 40, 400)
    pts = np.unique(np.column_stack([rows, cols]), axis=0)
    rows, cols = pts[:, 0].astype(np.int64), pts[:, 1].astype(np.int64)
    basin = ((rows // 7 + cols // 5) % 3 + 1).astype(np.int64)         # arbitrary basin pattern
    keep = {tuple(pts[i]) for i in select_block_samples(rows, cols, basin, 5, 30.0, 30.0)}
    for perm in (rng.permutation(rows.size), np.argsort(cols, kind="stable"), np.arange(rows.size)[::-1]):
        again = {(int(rows[perm][i]), int(cols[perm][i])) for i in select_block_samples(rows[perm], cols[perm], basin[perm], 5, 30.0, 30.0)}
        assert again == keep                                           # no dependence on the reading order
    by_key = {}
    for r, c, b in zip(rows, cols, basin):
        by_key.setdefault((r // 5, c // 5, b), []).append((r, c))
    assert len(keep) == len(by_key)                                    # exactly one per (block, basin)
    for (br, bc, b), cand in by_key.items():
        cr, cc = br * 5 + 2.0, bc * 5 + 2.0
        best = min(cand, key=lambda rc: ((rc[0] - cr) ** 2 + (rc[1] - cc) ** 2, rc[0], rc[1]))
        assert best in keep


# ---------------------------------------------------------------- sampling -----------
def make_net(tmp_path, with_hole=False):
    """20x30 raster: a trunk along row 10 (basin 1 west of col 15, basin 2 east) and a tributary column 8 (basin 3)."""
    h, w = 20, 30
    dem = (100 + np.add.outer(np.arange(h) * 0.5, np.arange(w) * 0.1)).astype("float32")
    drain = np.zeros((h, w), np.uint8); drain[10, :] = 1; drain[:10, 8] = 1
    basins = np.full((h, w), 2, np.int32); basins[:, :15] = 1; basins[:11, 5:12] = 3
    if with_hole:
        dem[10, 20] = NODATA
    d = write_dem(tmp_path / "dem.tif", dem)
    r = write_dem(tmp_path / "drain.tif", drain, nodata=255, dtype="uint8")
    b = write_dem(tmp_path / "basins.tif", basins, nodata=BN, dtype="int32")
    return d, r, b, dem, drain, basins


def test_sampling_interval_one_keeps_all_valid_drainage_cells_and_inherits_dem_elevation(tmp_path):
    d, r, b, dem, drain, basins = make_net(tmp_path, with_hole=True)
    table, stats = sample_drainage(r, d, b, 1)
    valid = (drain == 1) & (dem != NODATA)
    assert len(table) == int(valid.sum()) and stats["candidate_drainage_cells"] == len(table)   # the nodata cell is excluded
    for ba, ro, co, x, y, z in zip(table.basin, table.row, table.col, table.x, table.y, table.z):
        assert drain[ro, co] == 1 and z == pytest.approx(float(dem[ro, co])) and ba == basins[ro, co]
        assert x == pytest.approx(X0 + (co + 0.5) * RES) and y == pytest.approx(Y0 - (ro + 0.5) * RES)
    assert (table.z != NODATA).all()                                                            # nodata never becomes a sample


def test_sampling_larger_interval_reduces_samples_deterministically_and_keeps_every_network_basin(tmp_path, monkeypatch):
    d, r, b, *_ = make_net(tmp_path)
    t1, _ = sample_drainage(r, d, b, 1)
    t4, s4 = sample_drainage(r, d, b, 4)
    t4b, _ = sample_drainage(r, d, b, 4)
    assert 0 < len(t4) < len(t1) and np.array_equal(t4.row, t4b.row) and np.array_equal(t4.col, t4b.col)
    assert set(t4.basin.tolist()) == {1, 2, 3} == set(t1.basin.tolist())                        # no basin on the network is lost
    assert s4["basins_with_samples"] == s4["basins_intersecting_network"] == 3
    monkeypatch.setattr(rasters, "BLOCK_BYTES", 400)                                            # many stripes (aligned to 4 rows)
    t4c, _ = sample_drainage(r, d, b, 4)
    assert all(np.array_equal(getattr(t4, k), getattr(t4c, k)) for k in ("basin", "row", "col", "z"))


def test_sparse_and_tiny_networks(tmp_path):
    dem = np.full((9, 9), 5.0, "float32"); drain = np.zeros((9, 9), np.uint8); drain[4, 4] = 1     # a single drainage cell
    basins = np.full((9, 9), 1, np.int32)
    d = write_dem(tmp_path / "d.tif", dem); r = write_dem(tmp_path / "r.tif", drain, nodata=255, dtype="uint8")
    b = write_dem(tmp_path / "b.tif", basins, nodata=BN, dtype="int32")
    table, _ = sample_drainage(r, d, b, 10)
    assert len(table) == 1 and (table.row[0], table.col[0]) == (4, 4)


# ----------------------------------------------------------- surface per basin -------
def run_surface(tmp_path, d, b, table, power=2.0, name="s.tif"):
    w = WarningCollector()
    out = tmp_path / name
    stats = interpolate_surface(table, b, d, out, grid_of(d), power, w)
    with rasterio.open(out) as s:
        return s.read(1), s.nodata, stats, w


def reference_surface(dem, basins, table, power):
    ref = np.full(dem.shape, np.nan)
    rows, cols = np.nonzero((dem != NODATA) & (basins >= 1))
    for ro, co in zip(rows, cols):
        m = table.basin == basins[ro, co]
        if m.any():
            ref[ro, co] = manual_idw(X0 + (co + 0.5) * RES, Y0 - (ro + 0.5) * RES, table.x[m], table.y[m], table.z[m], power)
    return ref


def test_surface_matches_in_memory_reference_also_with_many_stripes(tmp_path, monkeypatch):
    d, r, b, dem, drain, basins = make_net(tmp_path)
    table, _ = sample_drainage(r, d, b, 2)
    ref = reference_surface(dem, basins, table, 2.0)
    s1, nd, stats, _ = run_surface(tmp_path, d, b, table)
    ok = ~np.isnan(ref)
    assert s1[ok] == pytest.approx(ref[ok].astype("float32"), rel=1e-6) and nd == NODATA
    assert stats["target_cells"] == int(ok.sum()) and stats["basins_interpolated"] == 3
    monkeypatch.setattr(rasters, "BLOCK_BYTES", 600)
    s2, *_ = run_surface(tmp_path, d, b, table, name="s2.tif")
    assert np.array_equal(s1, s2)                                                    # windowed == whole-raster result


def test_a_basin_depends_only_on_its_own_samples(tmp_path):
    d, r, b, dem, drain, basins = make_net(tmp_path)
    table, _ = sample_drainage(r, d, b, 1)
    base, *_ = run_surface(tmp_path, d, b, table)
    altered = SampleTable(table.basin.copy(), table.row, table.col, table.x, table.y, table.z.copy())
    altered.z[altered.basin == 1] += 5000.0                                         # drastic change in basin 1 only
    new, *_ = run_surface(tmp_path, d, b, altered, name="alt.tif")
    assert np.array_equal(new[basins == 2], base[basins == 2]) and np.array_equal(new[basins == 3], base[basins == 3])
    assert (new[basins == 1] > base[basins == 1] + 4000).all()
    # basin 2's surface is strictly inside the hull of its own samples' elevations (no influence from the 5000 m ones)
    z2 = table.z[table.basin == 2]
    assert base[basins == 2].min() >= z2.min() - 1e-3 and base[basins == 2].max() <= z2.max() + 1e-3


def test_cells_without_basin_or_dem_are_nodata_and_basins_without_samples_follow_the_policy(tmp_path):
    d, r, b, dem, drain, basins = make_net(tmp_path, with_hole=True)
    bas = basins.copy(); bas[0:3, 25:] = BN                                         # cells with no basin ID
    bas[15:, 0:6] = 9                                                               # basin 9: no drainage cell -> no samples
    b = write_dem(tmp_path / "b2.tif", bas, nodata=BN, dtype="int32")
    table, _ = sample_drainage(r, d, b, 1)
    surf, nd, stats, w = run_surface(tmp_path, d, b, table)
    assert (surf[0:3, 25:] == nd).all()                                             # no basin -> NoData
    assert surf[10, 20] == nd                                                       # DEM nodata -> NoData
    assert (surf[15:, 0:6] == nd).all()                                             # policy: NoData, never invented
    assert stats["basins_without_samples"] == 1 and stats["cells_in_basins_without_samples"] == 15 * 0 + int((bas[15:, 0:6] == 9).sum())
    assert any("no drainage sample" in m for m in w.messages)
    assert (surf[(bas >= 1) & (bas != 9) & (dem != NODATA)] != nd).all()


def test_nonfinite_or_missing_samples_never_produce_values(tmp_path):
    d, r, b, *_ = make_net(tmp_path)
    empty = SampleTable(*(np.zeros(0, dt) for dt in (np.int64, np.int64, np.int64, float, float, float)))
    surf, nd, stats, w = run_surface(tmp_path, d, b, empty)
    assert (surf == nd).all() and stats["interpolated_cells"] == 0


# ------------------------------------------------------------------- REM -------------
def test_rem_is_dem_minus_surface_preserves_nodata_and_negative_values(tmp_path):
    dem = np.array([[10, 12, NODATA], [8, 9, 10]], "float32")
    surf = np.array([[9, 13, 5], [NODATA, 9, 12]], "float32")
    d = write_dem(tmp_path / "d.tif", dem); s = write_dem(tmp_path / "s.tif", surf)
    w = WarningCollector()
    st = compute_rem(d, s, tmp_path / "rem.tif", grid_of(d), w)
    with rasterio.open(tmp_path / "rem.tif") as o:
        a = o.read(1)
        assert o.nodata == NODATA and o.dtypes[0] == "float32" and o.crs == rasterio.open(d).crs
    assert a[0, 0] == 1 and a[0, 1] == -1 and a[1, 1] == 0 and a[1, 2] == -2        # negatives are kept, not clipped to 0
    assert a[0, 2] == NODATA and a[1, 0] == NODATA
    assert st["valid_cells"] == 4 and st["negative_cells"] == 2 and st["cells_nodata_because_no_surface"] == 1
    assert st["min"] == -2 and st["max"] == 1


# ---------------------------------------------------------------- Subbasins ----------
class FakeBasinRunner:
    """Stands in for WhiteboxRunner.run_raster: records the call and writes a prepared basin raster."""

    def __init__(self, ids, nodata=BN):
        self.ids, self.nodata, self.calls = ids, nodata, []

    def run_raster(self, tool, flags, *, work_dir, output, normalize=True):
        self.calls.append((tool, list(flags), normalize))
        streams = [f.split("=", 1)[1] for f in flags if f.startswith("--streams=")][0]
        with rasterio.open(streams) as s:                      # the int32, safe-nodata copy of the drainage network
            self.streams = s.read(1); self.streams_nodata = s.nodata; self.streams_dtype = s.dtypes[0]
        write_dem(output, self.ids, nodata=self.nodata, dtype="int32")


def d8_pointer(tmp_path, shape):
    ptr = np.full(shape, 2, np.int16)
    return write_dem(tmp_path / "ptr.tif", ptr, nodata=-32768, dtype="int16")


def test_delineate_basins_passes_only_supported_flags_and_a_safe_streams_raster(tmp_path):
    d, r, b, dem, drain, basins = make_net(tmp_path)
    ptr = d8_pointer(tmp_path, dem.shape)
    ids = basins.copy()
    runner = FakeBasinRunner(ids)
    (tmp_path / "work").mkdir()
    st = delineate_basins(runner, ptr, r, tmp_path / "ids.tif", tmp_path / "work", grid_of(d), WarningCollector())
    tool, flags, normalize = runner.calls[0]
    assert tool == "Subbasins" and normalize is True
    assert [f.split("=")[0] for f in flags] == ["--d8_pntr", "--streams"]                # nothing invented
    assert runner.streams_dtype == "int32" and runner.streams_nodata == -9999            # never 255: IDs up to 255+ stay valid
    assert np.array_equal(runner.streams, drain.astype(np.int32))
    assert st["basins"] == 3 and st["cells_without_id"] == 0 and st["basins_without_drainage_cells"] == 0
    assert st["basin_cells_min"] > 0 and st["largest_basin_fraction"] == st["basin_cells_max"] / dem.size
    assert not (tmp_path / "work" / "rem_streams_input.tif").exists()


def test_delineate_basins_handles_a_single_unit_and_ids_above_255(tmp_path):
    d, r, b, dem, drain, basins = make_net(tmp_path)
    ptr = d8_pointer(tmp_path, dem.shape)
    one = np.full(dem.shape, 1, np.int32)
    st = delineate_basins(FakeBasinRunner(one), ptr, r, tmp_path / "one.tif", tmp_path, grid_of(d), WarningCollector())
    assert st["basins"] == 1
    big = (basins * 100 + 55).astype(np.int32)                                           # 155, 255, 355
    st = delineate_basins(FakeBasinRunner(big), ptr, r, tmp_path / "big.tif", tmp_path, grid_of(d), WarningCollector())
    assert st["basins"] == 3 and st["largest_id"] == 355 and st["ids_contiguous"] is False


@pytest.mark.parametrize("case, match", [
    ("nodata_is_an_id", "also a possible basin ID"),
    ("drain_without_id", "no basin ID"),
    ("all_nodata", "no valid basin units"),
])
def test_delineate_basins_rejects_incompatible_results(tmp_path, case, match):
    d, r, b, dem, drain, basins = make_net(tmp_path)
    ptr = d8_pointer(tmp_path, dem.shape)
    ids, nodata = basins.copy(), BN
    if case == "nodata_is_an_id":
        nodata = 255
    elif case == "drain_without_id":
        ids[10, 3] = BN
    else:
        ids[:] = BN
    with pytest.raises(ProcessingError, match=match):
        delineate_basins(FakeBasinRunner(ids, nodata), ptr, r, tmp_path / "x.tif", tmp_path, grid_of(d), WarningCollector())


def test_delineate_basins_reports_cells_that_drain_to_no_stream(tmp_path):
    d, r, b, dem, drain, basins = make_net(tmp_path)
    ptr = d8_pointer(tmp_path, dem.shape)
    ids = basins.copy(); ids[18:, 20:] = BN
    st = delineate_basins(FakeBasinRunner(ids), ptr, r, tmp_path / "x.tif", tmp_path, grid_of(d), WarningCollector())
    assert st["cells_without_id"] == 2 * 10 and st["cells_without_id_fraction"] == pytest.approx(20 / dem.size)
