from pathlib import Path

import pytest

from floodplain_delineation.config import load_config
from floodplain_delineation.exceptions import ConfigurationError


def write_config(tmp_path: Path, output_crs="EPSG:31983") -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
project:
  name: test
inputs:
  dem: dem.tif
  pilot_area:
    path: pilot.gpkg
    layer: pilot
crs:
  input: EPSG:4326
  output: {output_crs}
hydrology:
  processing_buffer:
    value: 1000
    units: m
  stream_threshold:
    value: 100
    units: cells
  depression_method: fill_depressions
floodplain:
  hand_threshold:
    value: 5
    units: m
outputs:
  directory: output
  conditioned_dem: conditioned_dem.tif
  flow_direction: flow_direction.tif
  flow_accumulation: flow_accumulation.tif
  drainage_raster: derived_drainage.tif
  drainage: derived_drainage.gpkg
  hand: hand.tif
  floodplain: potential_floodplain.tif
""",
        encoding="utf-8",
    )
    return config


def test_config_parses(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.stream_threshold.value == 100
    assert config.hand_threshold.value == 5
    assert config.processing_buffer.value == 1000
    assert config.output_crs.is_projected


def test_geographic_output_crs_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="projected"):
        load_config(write_config(tmp_path, "EPSG:4326"))


def test_missing_config_is_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")
