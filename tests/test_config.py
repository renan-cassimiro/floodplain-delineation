import copy

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
