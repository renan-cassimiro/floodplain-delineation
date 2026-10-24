import copy
from pathlib import Path

import pytest
import yaml

from floodplain_delineation.config import load_config, parse_config
from floodplain_delineation.exceptions import ConfigError


def test_valid_config_loads_and_applies_documented_defaults(base_cfg, tmp_path):
    del base_cfg["execution"]
    cfg = parse_config(base_cfg)
    assert cfg.project_name == "potential_floodplain_delineation"
    assert cfg.processing.conditioning_method == "breach"
    assert cfg.outputs.intermediates == "intermediate" and cfg.outputs.final == "final"
    assert cfg.execution.overwrite is False
    assert cfg.inputs.dem.is_absolute()
    assert cfg.to_dict()["processing"]["hand"]["threshold"] == 3.0


def test_load_from_yaml_file_records_hash(base_cfg, tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(base_cfg))
    cfg = load_config(f)
    assert len(cfg.source_sha256) == 64


def test_invalid_yaml_syntax(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("inputs: [unclosed\n  x: : :")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(f)


def test_empty_and_missing_file(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        load_config(f)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "path",
    [
        "inputs.dem", "inputs.pilot_area", "inputs.vertical_units", "processing.buffer_distance",
        "processing.conditioning", "processing.flow", "processing.drainage.threshold_type",
        "processing.drainage.threshold_value", "processing.hand.threshold", "outputs.directory",
    ],
)
def test_missing_required_parameter_is_rejected(base_cfg, path):
    cfg = copy.deepcopy(base_cfg)
    node = cfg
    *parents, leaf = path.split(".")
    for p in parents:
        node = node[p]
    del node[leaf]
    with pytest.raises(ConfigError, match="Missing required"):
        parse_config(cfg)


def test_missing_required_top_level_section(base_cfg):
    del base_cfg["processing"]
    with pytest.raises(ConfigError, match="Missing required"):
        parse_config(base_cfg)


@pytest.mark.parametrize(
    "dotted, value",
    [
        ("processing.conditioning.method", "magic"),
        ("processing.flow.algorithm", "dinf"),
        ("processing.drainage.threshold_type", "percent"),
        ("processing.drainage.threshold_value", -1),
        ("processing.drainage.threshold_value", 0),
        ("processing.drainage.threshold_value", 12.5),   # cells must be whole
        ("processing.hand.threshold", -0.1),
        ("processing.buffer_distance", -10),
        ("processing.buffer_units", "furlongs"),
        ("processing.hand.threshold", True),              # bool is not a number
        ("processing.hand.threshold", "5"),
        ("processing.hand.threshold", float("nan")),
        ("inputs.vertical_units", "parsecs"),
        ("execution.overwrite", "yes"),
    ],
)
def test_invalid_values_are_rejected(base_cfg, dotted, value):
    node = base_cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = value
    with pytest.raises(ConfigError):
        parse_config(base_cfg)


def test_area_threshold_may_be_fractional(base_cfg):
    base_cfg["processing"]["drainage"] = {"threshold_type": "area_km2", "threshold_value": 2.5}
    assert parse_config(base_cfg).processing.drainage_threshold_value == 2.5


def test_unknown_keys_rejected_at_any_level(base_cfg):
    base_cfg["processing"]["hand"]["thresh"] = 4
    with pytest.raises(ConfigError, match="Unknown parameter"):
        parse_config(base_cfg)
    base_cfg["processing"]["hand"].pop("thresh")
    base_cfg["extra"] = 1
    with pytest.raises(ConfigError, match="Unknown parameter"):
        parse_config(base_cfg)


def test_output_subdirs_must_be_relative_and_distinct(base_cfg):
    base_cfg["outputs"]["intermediates"] = "../escape"
    with pytest.raises(ConfigError, match="relative"):
        parse_config(copy.deepcopy(base_cfg))
    base_cfg["outputs"]["intermediates"] = "same"
    base_cfg["outputs"]["final"] = "same"
    with pytest.raises(ConfigError, match="different"):
        parse_config(base_cfg)


def test_section_must_be_mapping(base_cfg):
    base_cfg["inputs"] = ["dem.tif"]
    with pytest.raises(ConfigError, match="mapping"):
        parse_config(base_cfg)


# ------------------------------- execution options for large runs / resume -------------------------------
def test_execution_defaults_for_new_options(base_cfg):
    ex = parse_config(base_cfg).execution
    assert (ex.resume, ex.max_procs, ex.heartbeat_seconds, ex.whitebox_version, ex.scratch_directory) == (False, None, 300.0, None, None)


def test_execution_new_options_are_validated_and_echoed(base_cfg, tmp_path):
    base_cfg["execution"].update({"resume": True, "max_procs": 18, "heartbeat_seconds": 60,
                                  "whitebox_version": "2.4.0", "scratch_directory": str(tmp_path / "scratch")})
    cfg = parse_config(base_cfg)
    eff = cfg.to_dict()["execution"]
    assert eff["resume"] is True and eff["max_procs"] == 18 and eff["whitebox_version"] == "2.4.0"
    assert eff["scratch_directory"] == str((tmp_path / "scratch").resolve())


@pytest.mark.parametrize("key, value", [("max_procs", 0), ("max_procs", 2.5), ("max_procs", "all"), ("heartbeat_seconds", 0),
                                        ("resume", "yes"), ("whitebox_version", 2.4)])
def test_execution_new_options_reject_bad_values(base_cfg, key, value):
    base_cfg["execution"][key] = value
    with pytest.raises(ConfigError):
        parse_config(base_cfg)


def test_overwrite_and_resume_are_mutually_exclusive(base_cfg):
    base_cfg["execution"].update({"overwrite": True, "resume": True})
    with pytest.raises(ConfigError, match="cannot both be true"):
        parse_config(base_cfg)


def test_pilot_area_is_optional_and_path_typed(base_cfg):
    cfg = parse_config(base_cfg)
    assert isinstance(cfg.inputs.pilot_area, Path)
    base_cfg["inputs"]["pilot_area"] = None
    cfg = parse_config(base_cfg)
    assert cfg.inputs.pilot_area is None and cfg.to_dict()["inputs"]["pilot_area"] is None
