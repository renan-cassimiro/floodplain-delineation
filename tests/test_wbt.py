import os
import sys
from pathlib import Path

import pytest

from floodplain_delineation.exceptions import WhiteboxError
from floodplain_delineation.wbt import WhiteboxRunner, locate_executable

FAKE = Path(__file__).parent / "fake_wbt.py"


@pytest.fixture
def runner():
    return WhiteboxRunner(FAKE)


@pytest.mark.parametrize("mode, text", [("exit1", "exit status 101"), ("error_text", "reported an error"), ("no_output", "without creating")])
def test_failures_are_detected(runner, tmp_path, monkeypatch, mode, text):
    monkeypatch.setenv("FAKE_WBT_MODE", mode)
    out = tmp_path / "o.tif"
    with pytest.raises(WhiteboxError, match=text):
        runner.run("BreachDepressions", [f"--dem={tmp_path/'a.tif'}", f"--output={out}"], work_dir=tmp_path, outputs=[out])


def test_undocumented_flag_is_refused_before_running(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WBT_MODE", "ok")
    with pytest.raises(WhiteboxError, match="not documented"):
        runner.run("X", ["--bogus=1"], work_dir=tmp_path, outputs=[])


def test_version_and_recorded_calls(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_WBT_MODE", "ok")
    assert "FAKE" in runner.version()
    runner.run("X", ["--dem=a"], work_dir=tmp_path, outputs=[])
    assert runner.calls[-1]["tool"] == "X" and runner.calls[-1]["returncode"] == 0


def test_missing_executable_has_actionable_message(tmp_path):
    with pytest.raises(WhiteboxError, match="not found"):
        locate_executable(tmp_path / "nope")
