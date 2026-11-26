"""Runner behaviour added for large runs: streaming/heartbeat, max_procs, timeout, version policy, preflight."""

import logging
from pathlib import Path

import pytest

from conftest import config_dict
from floodplain_delineation import resources
from floodplain_delineation.config import parse_config
from floodplain_delineation.exceptions import WhiteboxError
from floodplain_delineation.logging_utils import WarningCollector
from floodplain_delineation.outputs import plan_outputs
from floodplain_delineation.pipeline import _check_wbt_version
from floodplain_delineation.preflight import run_preflight
from floodplain_delineation.wbt import WhiteboxRunner, parse_version

FAKE = Path(__file__).parent / "fake_wbt.py"


def test_heartbeat_and_streaming_are_logged_and_timings_recorded(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_WBT_MODE", "slow")
    monkeypatch.setenv("FAKE_WBT_SLEEP", "1.3")
    out = tmp_path / "o.tif"
    runner = WhiteboxRunner(FAKE, heartbeat_seconds=0.3)
    logger = logging.getLogger("floodplain_delineation")
    logger.propagate = True
    with caplog.at_level(logging.DEBUG, logger="floodplain_delineation"):
        runner.run("BreachDepressions", [f"--dem={tmp_path / 'a.tif'}", f"--output={out}"], work_dir=tmp_path, outputs=[out])
    text = caplog.text
    assert "still running" in text and "starting" in text and "finished in" in text   # heartbeat + streamed line
    call = runner.calls[-1]
    assert call["seconds"] >= 1.2 and call["returncode"] == 0


def test_timeout_kills_the_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WBT_MODE", "slow")
    monkeypatch.setenv("FAKE_WBT_SLEEP", "20")
    out = tmp_path / "o.tif"
    runner = WhiteboxRunner(FAKE, timeout=0.5, heartbeat_seconds=0.2)
    with pytest.raises(WhiteboxError, match="timeout"):
        runner.run("BreachDepressions", [f"--output={out}"], work_dir=tmp_path, outputs=[out])


@pytest.mark.parametrize("max_procs, expected", [(4, True), (None, False)])
def test_max_procs_flag_is_passed_only_when_configured(tmp_path, monkeypatch, max_procs, expected):
    args_file = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_WBT_MODE", "ok")
    monkeypatch.setenv("FAKE_WBT_ARGS_FILE", str(args_file))
    WhiteboxRunner(FAKE, max_procs=max_procs).run("X", ["--dem=a"], work_dir=tmp_path, outputs=[])
    line = args_file.read_text()
    assert ("--max_procs=4" in line) is expected


def test_parse_version():
    assert parse_version("WhiteboxTools v2.4.0 (c) Dr. John Lindsay 2017-2023") == "2.4.0"
    assert parse_version("WhiteboxTools vFAKE") is None


def test_version_pin_and_untested_version_warning():
    w = WarningCollector()
    _check_wbt_version("2.4.0", "WhiteboxTools v2.4.0 (c) ...", w)          # pinned and matching: silent
    assert len(w) == 0
    with pytest.raises(WhiteboxError, match="whitebox_version"):
        _check_wbt_version("2.4.0", "WhiteboxTools v2.0.0 by Dr. ...", w)
    _check_wbt_version(None, "WhiteboxTools v2.4.0 (c) ...", w)               # untested and unpinned: warns
    assert len(w) == 1 and "not covered" in w.messages[0]
    w2 = WarningCollector()
    _check_wbt_version(None, "WhiteboxTools v2.0.0 by Dr. ...", w2)           # tested: silent
    assert len(w2) == 0


def test_preflight_reports_and_warns(tmp_path, monkeypatch):
    cfg = parse_config(config_dict(tmp_path, tmp_path / "dem.tif", tmp_path / "p.gpkg", **{"execution.max_procs": 10_000}))
    plan = plan_outputs(cfg)
    monkeypatch.setattr(resources, "free_disk_bytes", lambda path: 1024)
    monkeypatch.setattr(resources, "available_memory_bytes",
                        lambda: {"machine_available": 2 ** 30, "cgroup_limit": None, "effective": 2 ** 30})
    w = WarningCollector()
    info = run_preflight(18993, 48316, plan, cfg, w)
    assert info["processing_window_cells"] == 18993 * 48316
    assert info["one_raster_float32_gib"] == pytest.approx(18993 * 48316 * 4 / 2 ** 30)
    assert info["disk_required_outputs_gib"] == pytest.approx(18993 * 48316 * 30 / 2 ** 30)
    joined = " ".join(w.messages)
    assert "Free disk space" in joined and "Available memory" in joined and "max_procs" in joined


def test_preflight_is_quiet_when_resources_are_ample(tmp_path, monkeypatch):
    cfg = parse_config(config_dict(tmp_path, tmp_path / "dem.tif", tmp_path / "p.gpkg"))
    monkeypatch.setattr(resources, "free_disk_bytes", lambda path: 2 ** 40)
    monkeypatch.setattr(resources, "available_memory_bytes",
                        lambda: {"machine_available": 2 ** 40, "cgroup_limit": None, "effective": 2 ** 40})
    w = WarningCollector()
    run_preflight(100, 100, plan_outputs(cfg), cfg, w)
    assert len(w) == 0


def test_resources_never_raise(tmp_path):
    assert resources.free_disk_bytes(tmp_path / "does" / "not" / "exist") is not None
    assert resources.process_rss_bytes(99999999) is None
    assert resources.format_duration(3725) == "1:02:05"
