"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pyproj import CRS

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class Quantity:
    value: float
    units: str


@dataclass(frozen=True)
class Config:
    project_name: str
    dem: Path
    pilot_area: Path
    pilot_layer: str
    input_crs: CRS
    output_crs: CRS
    processing_buffer: Quantity
    stream_threshold: Quantity
    hand_threshold: Quantity
    depression_method: str
    output_dir: Path
    outputs: dict[str, Path]


def _require(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration key: {path}.{key}")
    return mapping[key]


def _quantity(mapping: dict[str, Any], path: str, allowed_units: set[str]) -> Quantity:
    value = _require(mapping, "value", path)
    units = str(_require(mapping, "units", path)).lower()
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(f"{path}.value must be numeric.")
    if units not in allowed_units:
        raise ConfigurationError(
            f"{path}.units must be one of {sorted(allowed_units)}; got {units!r}."
        )
    if float(value) <= 0:
        raise ConfigurationError(f"{path}.value must be greater than zero.")
    return Quantity(float(value), units)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project = _require(raw, "project", "root")
    inputs = _require(raw, "inputs", "root")
    pilot = _require(inputs, "pilot_area", "inputs")
    crs_cfg = _require(raw, "crs", "root")
    hydrology = _require(raw, "hydrology", "root")
    floodplain = _require(raw, "floodplain", "root")
    outputs_cfg = _require(raw, "outputs", "root")

    try:
        input_crs = CRS.from_user_input(_require(crs_cfg, "input", "crs"))
        output_crs = CRS.from_user_input(_require(crs_cfg, "output", "crs"))
    except Exception as exc:
        raise ConfigurationError(f"Invalid CRS configuration: {exc}") from exc

    if not output_crs.is_projected:
        raise ConfigurationError("Output CRS must be projected and use metric units.")
    linear_units = {
        axis.unit_name.lower()
        for axis in output_crs.axis_info
        if axis.unit_name
    }
    if not any("metre" in unit or "meter" in unit for unit in linear_units):
        raise ConfigurationError(
            "Output CRS must use metres as its linear unit; a geographic CRS is not suitable."
        )

    config_dir = path.parent.parent if path.parent.name == "config" else path.parent
    dem = Path(_require(inputs, "dem", "inputs"))
    pilot_path = Path(_require(pilot, "path", "inputs.pilot_area"))
    if not dem.is_absolute():
        dem = (config_dir / dem).resolve()
    if not pilot_path.is_absolute():
        pilot_path = (config_dir / pilot_path).resolve()

    output_dir = Path(_require(outputs_cfg, "directory", "outputs"))
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()

    output_names = {
        key: str(value)
        for key, value in outputs_cfg.items()
        if key != "directory"
    }
    required_outputs = {
        "conditioned_dem",
        "flow_direction",
        "flow_accumulation",
        "drainage_raster",
        "drainage",
        "hand",
        "floodplain",
    }
    missing = required_outputs - set(output_names)
    if missing:
        raise ConfigurationError(
            f"Missing output names: {', '.join(sorted(missing))}"
        )

    outputs = {
        key: (Path(value) if Path(value).is_absolute() else output_dir / value)
        for key, value in output_names.items()
    }

    stream_threshold = _quantity(
        _require(hydrology, "stream_threshold", "hydrology"),
        "hydrology.stream_threshold",
        {"cells"},
    )
    processing_buffer = _quantity(
        _require(hydrology, "processing_buffer", "hydrology"),
        "hydrology.processing_buffer",
        {"m", "metre", "metres", "meter", "meters"},
    )
    hand_threshold = _quantity(
        _require(floodplain, "hand_threshold", "floodplain"),
        "floodplain.hand_threshold",
        {"m", "metre", "metres", "meter", "meters"},
    )

    depression_method = str(hydrology.get("depression_method", "fill_depressions"))
    if depression_method != "fill_depressions":
        raise ConfigurationError(
            "Only hydrology.depression_method='fill_depressions' is implemented in this version."
        )

    return Config(
        project_name=str(_require(project, "name", "project")),
        dem=dem,
        pilot_area=pilot_path,
        pilot_layer=str(_require(pilot, "layer", "inputs.pilot_area")),
        input_crs=input_crs,
        output_crs=output_crs,
        processing_buffer=processing_buffer,
        stream_threshold=stream_threshold,
        hand_threshold=hand_threshold,
        depression_method=depression_method,
        output_dir=output_dir,
        outputs=outputs,
    )
