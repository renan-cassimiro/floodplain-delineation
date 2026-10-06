import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from conftest import CRS, NODATA, RES, X0, Y0, config_dict, requires_wbt, write_dem, write_gpkg
from floodplain_delineation.cli import main
from floodplain_delineation.config import parse_config
from floodplain_delineation.exceptions import OutputError, ProcessingError, WhiteboxError
from floodplain_delineation.pipeline import run_pipeline

FAKE = str(Path(__file__).parent / "fake_wbt.py")


def make_cfg(tmp_path, dem, gpkg, **over):
    return parse_config(config_dict(tmp_path, dem, gpkg, **over))


# ---------------------- failure propagation (no WhiteboxTools needed) ---------------------- #
def test_failed_whitebox_operation_propagates_and_is_recorded(valley_dataset, tmp_path, monkeypatch):
    dem, gpkg, _ = valley_dataset
    monkeypatch.setenv("FAKE_WBT_MODE", "exit1")
    cfg = make_cfg(tmp_path, dem, gpkg, **{"execution.whitebox_executable": FAKE})
    with pytest.raises(WhiteboxError, match="exit status 101"):
        run_pipeline(cfg)
    out = tmp_path / "out"
    assert not (out / "final" / "potential_floodplain.tif").exists()      # no product after a failure
    summary = json.loads((out / "processing_summary.json").read_text())
    assert summary["status"] == "failed" and "exit status 101" in summary["error"]
    assert (out / "processing.log").read_text().count("Processing failed") == 1


def test_whitebox_error_is_a_processing_error():
    assert issubclass(WhiteboxError, ProcessingError)


def test_existing_outputs_block_run_without_overwrite_and_are_untouched(valley_dataset, tmp_path, monkeypatch):
    dem, gpkg, _ = valley_dataset
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "processing_summary.json"
    marker.write_text("PREVIOUS RESULT")
    cfg = make_cfg(tmp_path, dem, gpkg, **{"execution.whitebox_executable": FAKE})
    with pytest.raises(OutputError, match="already exist"):
        run_pipeline(cfg)
    assert marker.read_text() == "PREVIOUS RESULT"
    # with overwrite the run proceeds (and fails later because of the fake tool, replacing the summary)
    monkeypatch.setenv("FAKE_WBT_MODE", "exit1")
    cfg2 = make_cfg(tmp_path, dem, gpkg, **{"execution.whitebox_executable": FAKE, "execution.overwrite": True})
    with pytest.raises(WhiteboxError):
        run_pipeline(cfg2)
    assert json.loads(marker.read_text())["status"] == "failed"


def test_output_must_not_collide_with_inputs(valley_dataset, tmp_path):
    """A DEM that sits exactly where an output would be written must never be overwritten."""
    import shutil

    dem, gpkg, _ = valley_dataset
    clash = tmp_path / "out" / "intermediate" / "hand.tif"
    clash.parent.mkdir(parents=True)
    shutil.copy(dem, clash)
    cfg = make_cfg(tmp_path, clash, gpkg, **{"execution.overwrite": True})
    before = clash.read_bytes()
    with pytest.raises(OutputError, match="coincide with input"):
        run_pipeline(cfg)
    assert clash.read_bytes() == before


def test_cli_exit_codes(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 1
    assert "not found" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [")
    assert main(["--config", str(bad)]) == 1
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2


def test_cli_reports_missing_input_with_nonzero_status(base_cfg, tmp_path, capsys):
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(base_cfg))
    assert main(["--config", str(f)]) == 1
    assert "DEM file not found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()      # nothing created before inputs validate


# ------------------------------ integration (real WhiteboxTools) ------------------------------ #
def _pilot(arr):
    h, w = arr.shape
    from shapely.geometry import box
    return box(X0 + 30 * RES, Y0 - (h - 30) * RES, X0 + (w - 30) * RES, Y0 - 30 * RES)


def _read(path):
    with rasterio.open(path) as s:
        return s.read(1), s.nodata, s


@requires_wbt
@pytest.mark.wbt
def test_end_to_end_synthetic_workflow(valley_dataset, tmp_path):
    dem, gpkg, arr = valley_dataset
    cfg = make_cfg(tmp_path, dem, gpkg)
    summary = run_pipeline(cfg)
    out = tmp_path / "out"
    assert summary["status"] == "success"

    files = ["intermediate/conditioned_dem.tif", "intermediate/flow_direction.tif", "intermediate/flow_accumulation.tif",
             "intermediate/drainage_network.tif", "intermediate/hand.tif", "final/potential_floodplain.tif",
             "processing_summary.json", "processing.log", "processing_area.gpkg"]
    for f in files:
        assert (out / f).is_file(), f
    assert not (out / "intermediate" / ".wbt_tmp").exists()

    # all intermediates share the DEM-extent grid; resolution and CRS preserved (no resampling)
    ext_arr, _, ext = _read(out / "intermediate" / "dem_processing_extent.tif")
    with rasterio.open(dem) as src:
        assert ext.res == src.res and ext.crs == src.crs
        assert np.array_equal(ext_arr, src.read(1, window=rasterio.windows.Window(*summary["spatial"]["processing_window"])))
    for name in ["conditioned_dem", "flow_direction", "flow_accumulation", "drainage_network", "hand"]:
        _, _, s = _read(out / "intermediate" / f"{name}.tif")
        assert s.transform == ext.transform and s.shape == ext.shape and s.crs == ext.crs, name

    # drainage rule: accumulation >= effective cells
    acc, an, _ = _read(out / "intermediate" / "flow_accumulation.tif")
    drain, dn, _ = _read(out / "intermediate" / "drainage_network.tif")
    n = summary["stages"]["drainage"]["threshold"]["effective_cells"]
    assert n == 150 and np.array_equal(drain == 1, (acc >= n) & (acc != an))
    assert dn == 255 and drain.dtype == np.uint8

    # final mask semantics against the HAND raster
    hand, hn, _ = _read(out / "intermediate" / "hand.tif")
    mask, mn, ms = _read(out / "final" / "potential_floodplain.tif")
    assert mask.dtype == np.uint8 and mn == 255 and ms.res == ext.res and ms.crs == ext.crs
    c0 = round((ms.transform.c - ext.transform.c) / RES); r0 = round((ext.transform.f - ms.transform.f) / RES)
    h_sub = hand[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]]
    inside_valid = (mask != 255)
    assert np.array_equal(mask[inside_valid] == 1, (h_sub[inside_valid] <= 3.0))
    assert (mask[(drain[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] == 1) & inside_valid] == 1).all()

    s = json.loads((out / "processing_summary.json").read_text())
    for key in ["configuration", "spatial", "stages", "software_versions", "warnings", "limitations", "outputs",
                "started_utc", "finished_utc", "whitebox_calls"]:
        assert key in s, key
    assert s["configuration"]["effective"]["processing"]["hand"]["threshold"] == 3.0
    assert "whitebox_tools_executable" in s["software_versions"] and s["software_versions"]["rasterio"]
    assert s["stages"]["hand"]["min"] == pytest.approx(0.0)


@requires_wbt
@pytest.mark.wbt
def test_wbt_hand_matches_independent_d8_trace(valley_dataset, tmp_path):
    """HAND from ElevationAboveStream must equal HAND obtained by tracing the D8Pointer raster."""
    dem, gpkg, _ = valley_dataset
    run_pipeline(make_cfg(tmp_path, dem, gpkg))
    R = tmp_path / "out" / "intermediate"
    cond, cn, _ = _read(R / "conditioned_dem.tif"); ptr, _, _ = _read(R / "flow_direction.tif")
    drain, _, _ = _read(R / "drainage_network.tif"); hand, hn, _ = _read(R / "hand.tif")
    steps = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0), 16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}
    valid = cond != cn
    H, W = ptr.shape
    ref = np.full(ptr.shape, np.nan)
    for r in range(H):
        for c in range(W):
            if not valid[r, c]:
                continue
            rr, cc, seen = r, c, set()
            while (rr, cc) not in seen:
                seen.add((rr, cc))
                if drain[rr, cc] == 1:
                    ref[r, c] = cond[r, c] - cond[rr, cc]
                    break
                p = int(ptr[rr, cc])
                if p == 0:
                    break
                dr, dc = steps[p]
                rr, cc = rr + dr, cc + dc
                if not (0 <= rr < H and 0 <= cc < W) or not valid[rr, cc]:
                    break
    hv = hand != hn
    assert np.array_equal(hv, np.isfinite(ref))
    assert np.abs(hand[hv] - ref[hv]).max() < 1e-9


@requires_wbt
@pytest.mark.wbt
def test_d8_pointer_scheme_is_as_documented(tmp_path):
    from floodplain_delineation.wbt import WhiteboxRunner

    r = WhiteboxRunner()
    y, x = np.mgrid[0:20, 0:20].astype("float32")
    for name, plane, code in [("east", 100 - x, 2), ("south", 100 - y, 8), ("southeast", 100 - x - y, 4)]:
        p = write_dem(tmp_path / f"{name}.tif", plane)
        out = tmp_path / f"{name}_ptr.tif"
        r.run("D8Pointer", [f"--dem={p}", f"--output={out}"], work_dir=tmp_path, outputs=[out])
        arr, _, _ = _read(out)
        assert np.unique(arr[3:-3, 3:-3]).tolist() == [code], name


@requires_wbt
@pytest.mark.wbt
def test_save_flags_remove_unwanted_intermediates(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    cfg = make_cfg(tmp_path, dem, gpkg, **{"outputs.save_flow_direction": False, "outputs.save_hand": False})
    s = run_pipeline(cfg)
    out = tmp_path / "out"
    assert not (out / "intermediate" / "flow_direction.tif").exists()
    assert not (out / "intermediate" / "hand.tif").exists()
    assert (out / "final" / "potential_floodplain.tif").exists()
    assert sorted(s["intermediates_not_saved"]) == ["flow_direction", "hand"]


@requires_wbt
@pytest.mark.wbt
def test_cli_end_to_end_returns_zero(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(config_dict(tmp_path, dem, gpkg)))
    assert main(["--config", str(f)]) == 0
    assert (tmp_path / "out" / "final" / "potential_floodplain.tif").exists()


@requires_wbt
@pytest.mark.wbt
def test_fill_method_runs_or_fails_loudly_never_silently_corrupts(tmp_path):
    """DEM with an interior nodata hole: the fill tool may corrupt nodata; the pipeline must detect that."""
    from conftest import valley_terrain

    arr = valley_terrain()
    arr[50:54, 70:74] = NODATA
    dem = write_dem(tmp_path / "hole.tif", arr)
    gpkg = write_gpkg(tmp_path / "p.gpkg", [_pilot(arr)])
    cfg = make_cfg(tmp_path, dem, gpkg, **{"processing.conditioning.method": "fill"})
    try:
        run_pipeline(cfg)
    except ProcessingError as exc:
        assert "nodata" in str(exc).lower()
    else:  # tool handled the hole correctly: nodata must still be nodata in the final product
        out = tmp_path / "out" / "intermediate"
        c, cn, _ = _read(out / "conditioned_dem.tif"); e, en, _ = _read(out / "dem_processing_extent.tif")
        assert np.array_equal(c == cn, e == en)


@requires_wbt
@pytest.mark.wbt
def test_fill_method_on_clean_dem(valley_dataset, tmp_path):
    dem, gpkg, _ = valley_dataset
    s = run_pipeline(make_cfg(tmp_path, dem, gpkg, **{"processing.conditioning.method": "fill"}))
    assert s["status"] == "success" and s["stages"]["conditioning"]["tool"] == "FillDepressionsWangAndLiu"
