"""Resume: skip up-to-date stages, re-run only what changed, survive a mid-run failure (real WhiteboxTools)."""

import json

import numpy as np
import pytest
import rasterio

from conftest import config_dict, requires_wbt
from floodplain_delineation import rasters
from floodplain_delineation.config import parse_config
from floodplain_delineation.exceptions import ConfigError, OutputError, ProcessingError
from floodplain_delineation.pipeline import run_pipeline

pytestmark = [requires_wbt, pytest.mark.wbt]

ALL_STAGES = ["dem_extraction", "conditioning", "flow_direction", "flow_accumulation", "drainage", "hand"]


def cfg_of(tmp_path, dem, gpkg, **over):
    return parse_config(config_dict(tmp_path, dem, gpkg, **over))


def tools_run(summary):
    return [c["tool"] for c in summary["whitebox_calls"]]


def test_second_run_without_resume_or_overwrite_is_refused_with_a_hint(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    run_pipeline(cfg_of(tmp_path, dem, gpkg))
    with pytest.raises(OutputError, match="execution.resume"):
        run_pipeline(cfg_of(tmp_path, dem, gpkg))


def test_identical_rerun_reuses_everything_and_reproduces_the_result(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    s1 = run_pipeline(cfg_of(tmp_path, dem, gpkg))
    out = tmp_path / "out"
    before = (out / "final" / "potential_floodplain.tif").read_bytes()
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True}))
    assert s2["status"] == "success" and s2["stages_reused_from_previous_run"] == ALL_STAGES
    assert tools_run(s2) == []                                         # no WhiteboxTools call at all
    assert (out / "final" / "potential_floodplain.tif").read_bytes() == before
    for key in ("dem_extraction", "conditioning", "drainage", "hand", "flow_routing"):
        assert s2["stages"][key] == s1["stages"][key], key             # summary stays complete
    assert "Resume mode" in (out / "processing.log").read_text()
    assert (out / "processing.log").read_text().count("=== potential_floodplain_delineation ===") == 2   # log appended


def test_changing_only_the_hand_threshold_reruns_only_the_final_stage(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    s1 = run_pipeline(cfg_of(tmp_path, dem, gpkg))
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True, "processing.hand.threshold": 6}))
    assert tools_run(s2) == [] and s2["stages_reused_from_previous_run"] == ALL_STAGES
    assert s2["stages"]["floodplain"]["floodplain_cells"] > s1["stages"]["floodplain"]["floodplain_cells"]
    with rasterio.open(tmp_path / "out" / "final" / "potential_floodplain.tif") as m, \
            rasterio.open(tmp_path / "out" / "intermediate" / "hand.tif") as h:
        assert m.tags()["HAND_THRESHOLD"] == "6.0" and h.shape[0] >= m.shape[0]


def test_changing_the_drainage_threshold_reruns_drainage_and_hand_only(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    run_pipeline(cfg_of(tmp_path, dem, gpkg))
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True, "processing.drainage.threshold_value": 400}))
    assert tools_run(s2) == ["ExtractStreams", "ElevationAboveStream"]
    assert s2["stages_reused_from_previous_run"] == ["dem_extraction", "conditioning", "flow_direction", "flow_accumulation"]
    assert s2["stages"]["drainage"]["threshold"]["effective_cells"] == 400


def test_changing_the_conditioning_method_reruns_everything_downstream(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    run_pipeline(cfg_of(tmp_path, dem, gpkg))
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True, "processing.conditioning.method": "fill"}))
    assert s2["stages_reused_from_previous_run"] == ["dem_extraction"]
    assert tools_run(s2)[0] == "FillDepressionsWangAndLiu" and len(tools_run(s2)) == 5


def test_modified_dem_invalidates_the_cache(valley_dataset, tmp_path):
    dem, gpkg, arr = valley_dataset
    run_pipeline(cfg_of(tmp_path, dem, gpkg))
    with rasterio.open(dem, "r+") as d:
        d.write((arr + 1).astype("float32"), 1)
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True}))
    assert s2["stages_reused_from_previous_run"] == [] and len(tools_run(s2)) == 5


def test_tampered_output_is_detected_and_recomputed(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    s1 = run_pipeline(cfg_of(tmp_path, dem, gpkg))
    f = tmp_path / "out" / "intermediate" / "hand.tif"
    f.write_bytes(f.read_bytes()[:-1000])                                  # truncated file -> size mismatch
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True}))
    assert tools_run(s2) == ["ElevationAboveStream"]                       # only the damaged stage is recomputed
    assert s2["stages"]["hand"] == s1["stages"]["hand"]


def test_deleted_intermediate_is_not_recomputed_when_downstream_is_current(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"outputs.save_flow_direction": False, "outputs.save_flow_accumulation": False}))
    out = tmp_path / "out" / "intermediate"
    assert not (out / "flow_direction.tif").exists()
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True, "processing.hand.threshold": 5,
                                                     "outputs.save_flow_direction": False, "outputs.save_flow_accumulation": False}))
    assert tools_run(s2) == [] and s2["status"] == "success"


def test_failure_then_resume_continues_from_the_failed_stage(valley_dataset, tmp_path, monkeypatch):
    import floodplain_delineation.pipeline as pl

    dem, gpkg, _ = valley_dataset
    real = pl.compute_flow_accumulation

    def boom(*a, **k):
        raise ProcessingError("simulated failure after conditioning and flow direction")

    monkeypatch.setattr(pl, "compute_flow_accumulation", boom)
    with pytest.raises(ProcessingError, match="simulated"):
        run_pipeline(cfg_of(tmp_path, dem, gpkg))
    out = tmp_path / "out"
    failed = json.loads((out / "processing_summary.json").read_text())
    assert failed["status"] == "failed" and set(failed["stages"]) >= {"dem_extraction", "conditioning", "flow_routing"}
    assert "execution.resume: true" in (out / "processing.log").read_text()
    manifest = json.loads((out / "stage_manifest.json").read_text())["stages"]
    assert set(manifest) == {"dem_extraction", "conditioning", "flow_direction"}

    monkeypatch.setattr(pl, "compute_flow_accumulation", real)
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.resume": True}))
    assert s2["status"] == "success"
    assert s2["stages_reused_from_previous_run"] == ["dem_extraction", "conditioning", "flow_direction"]
    assert tools_run(s2) == ["D8FlowAccumulation", "ExtractStreams", "ElevationAboveStream"]


def test_overwrite_starts_from_scratch_and_clears_the_manifest(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    run_pipeline(cfg_of(tmp_path, dem, gpkg))
    s2 = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.overwrite": True}))
    assert s2["stages_reused_from_previous_run"] == [] and len(tools_run(s2)) == 5


def test_leftovers_of_an_interrupted_run_are_removed(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    inter = tmp_path / "out" / "intermediate"
    (inter / ".wbt_tmp").mkdir(parents=True)
    (inter / ".wbt_tmp" / "conditioned_dem.wbt_raw.tif").write_bytes(b"x" * 10)
    (inter / "conditioned_dem.tif.partial").write_bytes(b"x" * 10)
    run_pipeline(cfg_of(tmp_path, dem, gpkg))
    assert not (inter / "conditioned_dem.tif.partial").exists() and not (inter / ".wbt_tmp").exists()


def test_scratch_directory_receives_raw_outputs_and_is_cleaned(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    scratch = tmp_path / "scratch"
    s = run_pipeline(cfg_of(tmp_path, dem, gpkg, **{"execution.scratch_directory": str(scratch)}))
    assert s["status"] == "success" and not any(scratch.rglob("*.tif"))
    assert not (tmp_path / "out" / "intermediate" / ".wbt_tmp").exists()


def test_whole_pipeline_gives_identical_results_with_tiny_blocks(valley_dataset, tmp_path, monkeypatch):
    dem, gpkg, _ = valley_dataset
    a = run_pipeline(cfg_of(tmp_path / "a", dem, gpkg))
    monkeypatch.setattr(rasters, "BLOCK_BYTES", 1500)           # many stripes everywhere
    b = run_pipeline(cfg_of(tmp_path / "b", dem, gpkg))
    for key in ("dem_extraction", "conditioning", "drainage", "floodplain"):
        sa, sb = dict(a["stages"][key]), dict(b["stages"][key])
        for d in (sa, sb):
            d.pop("path", None)
        assert sa == sb, key
    for name in ("conditioned_dem", "flow_direction", "flow_accumulation", "drainage_network", "hand"):
        with rasterio.open(tmp_path / "a" / "out" / "intermediate" / f"{name}.tif") as x, \
                rasterio.open(tmp_path / "b" / "out" / "intermediate" / f"{name}.tif") as y:
            assert np.array_equal(x.read(1), y.read(1)), name
    with rasterio.open(tmp_path / "a" / "out" / "final" / "potential_floodplain.tif") as x, \
            rasterio.open(tmp_path / "b" / "out" / "final" / "potential_floodplain.tif") as y:
        assert np.array_equal(x.read(1), y.read(1))


def test_hand_statistics_match_numpy(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    s = run_pipeline(cfg_of(tmp_path, dem, gpkg))
    with rasterio.open(tmp_path / "out" / "intermediate" / "hand.tif") as h:
        a = h.read(1); v = a[a != h.nodata].astype("float64")
    st = s["stages"]["hand"]
    bw = (v.max() - v.min()) / 100_000
    assert st["valid_cells"] == v.size and st["min"] == pytest.approx(v.min()) and st["max"] == pytest.approx(v.max())
    assert st["mean"] == pytest.approx(v.mean()) and st["std"] == pytest.approx(v.std())
    for key, q in (("p05", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95)):
        lo, hi = np.percentile(v, q, method="lower"), np.percentile(v, q, method="higher")
        assert lo - 2 * bw - 1e-9 <= st[key] <= hi + 2 * bw + 1e-9, key      # histogram quantile within bin resolution


def test_performance_and_preflight_are_recorded(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    s = run_pipeline(cfg_of(tmp_path, dem, gpkg))
    perf = s["performance"]
    assert set(perf["stages"]) == set(ALL_STAGES) | {"floodplain"} and perf["total_seconds"] > 0
    assert all(c["seconds"] >= 0 and "peak_rss_gib_sampled" in c for c in s["whitebox_calls"])
    assert s["preflight"]["processing_window_cells"] > 0
