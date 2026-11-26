"""REM through the whole pipeline with the real WhiteboxTools: products, HAND untouched, resume invalidation."""

import numpy as np
import pytest
import rasterio

from conftest import NODATA, RES, X0, Y0, box, config_dict, requires_wbt, valley_terrain, write_dem, write_gpkg
from floodplain_delineation import rasters
from floodplain_delineation.config import parse_config
from floodplain_delineation.pipeline import run_pipeline
from floodplain_delineation.rem import load_samples
from floodplain_delineation.wbt import WhiteboxRunner

pytestmark = [requires_wbt, pytest.mark.wbt]
REM_ON = {"enabled": True, "sampling_interval_pixels": 3, "idw_power": 2.0}
REM_STAGES = ["basins", "rem_samples", "rem_surface", "rem"]
HAND_STAGES = ["dem_reprojection", "conditioning", "flow_direction", "flow_accumulation", "drainage", "hand"]


@pytest.fixture
def dataset(tmp_path):
    arr = valley_terrain()
    arr[20:24, 40:44] = NODATA                                   # a nodata hole inside the processing window, away from the valley
    dem = write_dem(tmp_path / "dem.tif", arr)
    h, w = arr.shape
    gpkg = write_gpkg(tmp_path / "p.gpkg", [box(X0 + 30 * RES, Y0 - (h - 30) * RES, X0 + (w - 30) * RES, Y0 - 30 * RES)])
    return dem, gpkg, arr


def cfg_of(tmp_path, dem, gpkg, rem=REM_ON, **over):
    return parse_config(config_dict(tmp_path, dem, gpkg, **{"processing.rem": rem, **over}))


def tools(summary):
    return [c["tool"] for c in summary["whitebox_calls"]]


def read(path):
    with rasterio.open(path) as r:
        return r.read(1), r.nodata, r


def test_rem_products_match_an_independent_brute_force_computation(dataset, tmp_path):
    dem, gpkg, arr = dataset
    s = run_pipeline(cfg_of(tmp_path, dem, gpkg))
    out = tmp_path / "out"
    ref_dem, dnd, rdem = read(out / "intermediate" / "dem_processing_crs.tif")
    basins, bnd, _ = read(out / "intermediate" / "basins.tif")
    surf, snd, rs = read(out / "rem" / "rem_drainage_surface.tif")
    rem, rnd, rr = read(out / "rem" / "rem.tif")
    drain, _, _ = read(out / "intermediate" / "drainage_network.tif")
    table = load_samples(out / "intermediate" / "rem_samples.npz")

    # same grid / CRS / transform / nodata convention in every product
    for r in (rs, rr):
        assert (r.crs, r.transform, r.shape) == (rdem.crs, rdem.transform, rdem.shape) and r.dtypes[0] == "float32"
    # the reference DEM is the unconditioned DEM on the processing grid: identical to the input window
    c0, r0, ww, hh = s["spatial"]["source_window"]
    win = arr[r0:r0 + hh, c0:c0 + ww]
    assert ref_dem.shape == win.shape and np.array_equal(ref_dem[ref_dem != dnd], win[win != NODATA])

    # samples: every one is a valid, drainage, in-basin cell, elevation taken from the reference DEM
    assert len(table) > 0 and (drain[table.row, table.col] == 1).all()
    assert np.array_equal(table.z, ref_dem[table.row, table.col].astype("float64")) and (basins[table.row, table.col] == table.basin).all()

    # brute force IDW per basin (all samples of the basin only) and REM = DEM - surface
    expect = np.full(ref_dem.shape, np.nan)
    for b in np.unique(basins[basins >= 1]):
        m = table.basin == b
        rows, cols = np.nonzero((basins == b) & (ref_dem != dnd))
        t = rdem.transform                                                  # real cell centres (the window has an offset)
        x, y = t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e
        for k in range(rows.size):
            d = np.hypot(x[k] - table.x[m], y[k] - table.y[m])
            expect[rows[k], cols[k]] = table.z[m][np.argmin(d)] if d.min() == 0 else (table.z[m] / d ** 2).sum() / (1 / d ** 2).sum()
    ok = ~np.isnan(expect)
    assert ((surf != snd) == ok).all()
    assert surf[ok] == pytest.approx(expect[ok].astype("float32"), rel=1e-6, abs=1e-4)
    assert rem[ok] == pytest.approx((ref_dem[ok].astype("float64") - surf[ok].astype("float64")).astype("float32"), abs=1e-4)
    assert ((rem != rnd) == ok).all()
    assert ((ref_dem == dnd) & (rem != rnd)).sum() == 0                        # every DEM nodata cell (incl. the hole) stays nodata
    assert (win == NODATA).any() and ((win == NODATA) <= (rem == rnd)).all()
    # exclusive, non-nested units: sizes + cells without ID == valid cells; every unit has a drainage cell
    st = s["stages"]["basins"]
    assert st["cells_with_id"] + st["cells_without_id"] == int((ref_dem != dnd).sum())
    assert set(np.unique(basins[drain == 1]).tolist()) == set(np.unique(basins[basins >= 1]).tolist()) and st["basins_without_drainage_cells"] == 0
    assert s["rem"]["enabled"] and s["rem"]["basins"] == st["basins"] and s["rem"]["samples"] == len(table)
    assert s["stages"]["rem"]["valid_cells"] == int(ok.sum())


def test_rem_never_changes_hand_or_the_floodplain_mask(dataset, tmp_path):
    dem, gpkg, _ = dataset
    off = run_pipeline(cfg_of(tmp_path / "off", dem, gpkg, rem={"enabled": False}))
    on = run_pipeline(cfg_of(tmp_path / "on", dem, gpkg))
    assert "Subbasins" not in tools(off) and tools(on).count("Subbasins") == 1
    assert off["rem"] == {"enabled": False} and not (tmp_path / "off" / "out" / "rem").exists()
    for rel in ("intermediate/hand.tif", "intermediate/conditioned_dem.tif", "intermediate/drainage_network.tif",
                "final/potential_floodplain.tif"):
        a, _, _ = read(tmp_path / "off" / "out" / rel); b, _, _ = read(tmp_path / "on" / "out" / rel)
        assert np.array_equal(a, b), rel
    strip = lambda d: {k: v for k, v in d.items() if k != "path"}  # noqa: E731
    assert off["stages"]["hand"] == on["stages"]["hand"] and strip(off["stages"]["floodplain"]) == strip(on["stages"]["floodplain"])


def test_rem_is_independent_of_the_memory_budget(dataset, tmp_path, monkeypatch):
    dem, gpkg, _ = dataset
    run_pipeline(cfg_of(tmp_path / "a", dem, gpkg))
    monkeypatch.setattr(rasters, "BLOCK_BYTES", 1500)
    run_pipeline(cfg_of(tmp_path / "b", dem, gpkg))
    for rel in ("intermediate/basins.tif", "rem/rem_drainage_surface.tif", "rem/rem.tif"):
        a, _, _ = read(tmp_path / "a" / "out" / rel); b, _, _ = read(tmp_path / "b" / "out" / rel)
        assert np.array_equal(a, b), rel


# ----------------------------------------------------------------------- resume / invalidation ----------------
def first(dataset, tmp_path, **over):
    dem, gpkg, _ = dataset
    return run_pipeline(cfg_of(tmp_path, dem, gpkg, **over))


def again(dataset, tmp_path, rem=REM_ON, **over):
    dem, gpkg, _ = dataset
    return run_pipeline(cfg_of(tmp_path, dem, gpkg, rem=rem, **{"execution.resume": True, **over}))


def test_identical_rerun_reuses_every_stage_including_the_rem(dataset, tmp_path):
    s1 = first(dataset, tmp_path)
    s2 = again(dataset, tmp_path)
    assert s2["stages_reused_from_previous_run"] == HAND_STAGES + REM_STAGES and tools(s2) == []
    for k in REM_STAGES:
        assert s2["stages"][k] == s1["stages"][k]


def test_changing_the_sampling_interval_reruns_only_the_rem_sampling_chain(dataset, tmp_path):
    first(dataset, tmp_path)
    s2 = again(dataset, tmp_path, rem={**REM_ON, "sampling_interval_pixels": 5})
    assert s2["stages_reused_from_previous_run"] == HAND_STAGES + ["basins"] and tools(s2) == []     # no WhiteboxTools needed
    assert {k for k, v in s2["performance"]["stages"].items() if v["status"] == "run"} == {"rem_samples", "rem_surface", "rem", "floodplain"}


def test_changing_the_idw_power_reruns_only_surface_and_rem(dataset, tmp_path):
    first(dataset, tmp_path)
    s2 = again(dataset, tmp_path, rem={**REM_ON, "idw_power": 3.0})
    assert s2["stages_reused_from_previous_run"] == HAND_STAGES + ["basins", "rem_samples"] and tools(s2) == []
    assert {k for k, v in s2["performance"]["stages"].items() if v["status"] == "run"} == {"rem_surface", "rem", "floodplain"}


def test_changing_the_drainage_threshold_invalidates_basins_and_everything_after_it(dataset, tmp_path):
    first(dataset, tmp_path)
    s2 = again(dataset, tmp_path, **{"processing.drainage.threshold_value": 400})
    assert tools(s2) == ["ExtractStreams", "Subbasins", "ElevationAboveStream"] or sorted(tools(s2)) == sorted(["ExtractStreams", "ElevationAboveStream", "Subbasins"])
    assert s2["stages_reused_from_previous_run"] == ["dem_reprojection", "conditioning", "flow_direction", "flow_accumulation"]


def test_changing_the_hand_threshold_does_not_touch_the_rem(dataset, tmp_path):
    first(dataset, tmp_path)
    s2 = again(dataset, tmp_path, **{"processing.hand.threshold": 6})
    assert tools(s2) == [] and s2["stages_reused_from_previous_run"] == HAND_STAGES + REM_STAGES


def test_changing_the_processing_crs_invalidates_everything(dataset, tmp_path):
    first(dataset, tmp_path)
    s2 = again(dataset, tmp_path, **{"processing.crs": "ESRI:102033"})
    assert s2["stages_reused_from_previous_run"] == [] and "Subbasins" in tools(s2) and len(tools(s2)) == 6
    assert s2["spatial"]["processing_crs"]["authority_code"] == "ESRI:102033"


def test_changing_the_whitebox_version_invalidates_every_whitebox_stage(dataset, tmp_path, monkeypatch):
    first(dataset, tmp_path)
    monkeypatch.setattr(WhiteboxRunner, "version", lambda self: "WhiteboxTools v9.9.9 (test)")
    s2 = again(dataset, tmp_path)
    assert s2["stages_reused_from_previous_run"] == ["dem_reprojection"] and len(tools(s2)) == 6


def test_changing_a_conditioning_parameter_invalidates_conditioning_and_downstream(dataset, tmp_path):
    lc = {"processing.conditioning": {"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 20}}}
    first(dataset, tmp_path, **lc)
    s2 = again(dataset, tmp_path, **lc)
    assert tools(s2) == []
    lc2 = {"processing.conditioning": {"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 25}}}
    s3 = again(dataset, tmp_path, **lc2)
    assert s3["stages_reused_from_previous_run"] == ["dem_reprojection"] and len(tools(s3)) == 6


def test_enabling_the_rem_later_runs_only_the_rem_stages(dataset, tmp_path):
    first(dataset, tmp_path, rem={"enabled": False})
    s2 = again(dataset, tmp_path)
    assert tools(s2) == ["Subbasins"] and s2["stages_reused_from_previous_run"] == HAND_STAGES
    assert s2["rem"]["enabled"] and (tmp_path / "out" / "rem" / "rem.tif").is_file()


# ----------------------------------------------------------------------- global CRS -------------------------
def test_pipeline_in_the_global_crs_keeps_every_product_on_one_grid(dataset, tmp_path):
    dem, gpkg, _ = dataset
    s = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"processing.crs": "ESRI:102033", "processing.target_resolution": 30.0}))
    out = tmp_path / "out"
    grids = {}
    for rel in ("intermediate/dem_processing_crs.tif", "intermediate/conditioned_dem.tif", "intermediate/flow_direction.tif",
                "intermediate/flow_accumulation.tif", "intermediate/drainage_network.tif", "intermediate/hand.tif",
                "intermediate/basins.tif", "rem/rem_drainage_surface.tif", "rem/rem.tif"):
        _, _, r = read(out / rel)
        grids[rel] = (r.crs.to_string()[:12], r.transform, r.shape)
        assert r.crs.to_wkt() == read(out / "intermediate/dem_processing_crs.tif")[2].crs.to_wkt()
    assert len(set(grids.values())) == 1
    sp = s["spatial"]
    assert sp["processing_crs"]["authority_code"] == "ESRI:102033" and sp["resolution"]["target_resolution_crs_units"] == [30.0, 30.0]
    assert sp["cell_area_km2"] == pytest.approx(30 * 30 / 1e6, rel=1e-9)               # equal-area CRS: exact cell area
    assert s["stages"]["dem_reprojection"]["resampling"] == "bilinear"
    assert s["stages"]["rem"]["valid_cells"] > 0
