"""YAML configuration loading and strict validation.

Design rules
------------
* Scientific parameters (buffer, conditioning method, flow algorithm, drainage
  threshold, HAND threshold, vertical units) are REQUIRED. No silent defaults.
  The processing CRS/resolution/resampling and the REM options have DOCUMENTED defaults
  (ESRI:102033, native-derived resolution, bilinear, REM disabled); when the REM is
  enabled, its sampling interval is required.
* Non-scientific parameters (project name, subdirectory names, save flags,
  overwrite/verbose) have documented defaults.
* Unknown keys are rejected so typos never pass silently.
* Relative paths are resolved against the current working directory (not the
  location of the YAML file) and stored as absolute paths for reproducibility.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from .exceptions import ConfigError, InputError
from .spatial import resolve_processing_crs

VERTICAL_UNITS = ("meters", "centimeters", "decimeters", "feet", "us_survey_feet")
BUFFER_UNITS = ("meters", "cells")
CONDITIONING_METHODS = ("breach", "fill", "breach_least_cost")
FLOW_ALGORITHMS = ("d8",)
DRAINAGE_THRESHOLD_TYPES = ("cells", "area_km2")
DEM_RESAMPLING_METHODS = ("near", "bilinear", "cubic", "cubic_spline", "lanczos")  # continuous data only

DEFAULT_PROCESSING_CRS = "ESRI:102033"  # South America Albers Equal Area Conic (ESRI authority, not EPSG)
DEFAULT_IDW_POWER = 2.0

DEFAULT_PROJECT_NAME = "potential_floodplain_delineation"


@dataclass(frozen=True)
class InputsConfig:
    dem: Path
    pilot_area: Optional[Path]   # None: the whole DEM footprint is processed
    pilot_layer: Optional[str]
    vertical_units: str


@dataclass(frozen=True)
class RemConfig:
    """Relative Elevation Model: DEM minus an IDW surface of drainage elevations, per basin."""

    enabled: bool = False
    sampling_interval_pixels: Optional[int] = None   # required when enabled
    idw_power: float = DEFAULT_IDW_POWER


@dataclass(frozen=True)
class BreachLeastCostConfig:
    """Parameters of WhiteboxTools ``BreachDepressionsLeastCost`` (method ``breach_least_cost``).

    ``max_distance_cells`` is the tool's ``--dist`` (maximum search distance of a breach path, in cells) and has
    no default: it bounds how far a breach may reach and therefore how much the terrain can be modified.
    """

    max_distance_cells: int
    max_cost: Optional[float] = None           # --max_cost (None: the tool's default, unbounded)
    minimize_distance: bool = False            # --min_dist (flag): prefer shorter breach channels
    flat_increment: Optional[float] = None     # --flat_increment (None: the tool's default)


@dataclass(frozen=True)
class ProcessingConfig:
    buffer_distance: float
    buffer_units: str
    conditioning_method: str
    flow_algorithm: str
    drainage_threshold_type: str
    drainage_threshold_value: float
    hand_threshold: float
    crs: str = DEFAULT_PROCESSING_CRS                # processing CRS of the whole pipeline (projected)
    target_resolution: Optional[float] = None        # CRS units; None: derived from the source DEM
    dem_resampling: str = "bilinear"
    rem: RemConfig = field(default_factory=RemConfig)
    breach_least_cost: Optional[BreachLeastCostConfig] = None   # required iff conditioning_method == breach_least_cost
    fill_remaining: bool = False                                # breach_least_cost only: --fill (breach, then fill what is left)


@dataclass(frozen=True)
class OutputsConfig:
    directory: Path
    intermediates: str
    final: str
    save_conditioned_dem: bool
    save_flow_direction: bool
    save_flow_accumulation: bool
    save_drainage_network: bool
    save_hand: bool
    rem: str = "rem"                    # sub-directory of the REM products (rem.tif, surface)
    save_basins: bool = True
    save_rem_samples: bool = True
    save_rem_drainage_surface: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    overwrite: bool
    verbose: bool
    whitebox_executable: Optional[Path]
    resume: bool = False                       # reuse up-to-date stage outputs of a previous run
    max_procs: Optional[int] = None            # passed to WhiteboxTools as --max_procs (None: tool default)
    heartbeat_seconds: float = 300.0           # progress message interval for long WhiteboxTools runs
    whitebox_version: Optional[str] = None     # exact WhiteboxTools version required (e.g. "2.4.0")
    scratch_directory: Optional[Path] = None   # where raw WhiteboxTools outputs are written (fast local disk)


@dataclass(frozen=True)
class Config:
    project_name: str
    inputs: InputsConfig
    processing: ProcessingConfig
    outputs: OutputsConfig
    execution: ExecutionConfig
    source_path: Optional[Path] = None
    source_sha256: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Effective (validated, defaults applied) configuration, JSON-friendly."""
        p, o = self.processing, self.outputs
        return {
            "project": {"name": self.project_name},
            "inputs": {
                "dem": str(self.inputs.dem),
                "pilot_area": str(self.inputs.pilot_area) if self.inputs.pilot_area else None,
                "pilot_layer": self.inputs.pilot_layer,
                "vertical_units": self.inputs.vertical_units,
            },
            "processing": {
                "buffer_distance": p.buffer_distance,
                "buffer_units": p.buffer_units,
                "conditioning": {
                    "method": p.conditioning_method,
                    **(
                        {
                            "breach_least_cost": {
                                "max_distance_cells": p.breach_least_cost.max_distance_cells,
                                "max_cost": p.breach_least_cost.max_cost,
                                "minimize_distance": p.breach_least_cost.minimize_distance,
                                "flat_increment": p.breach_least_cost.flat_increment,
                            },
                            "fill_remaining": p.fill_remaining,
                        }
                        if p.breach_least_cost is not None
                        else {}
                    ),
                },
                "flow": {"algorithm": p.flow_algorithm},
                "drainage": {
                    "threshold_type": p.drainage_threshold_type,
                    "threshold_value": p.drainage_threshold_value,
                },
                "hand": {"threshold": p.hand_threshold},
                "crs": p.crs,
                "target_resolution": p.target_resolution,
                "dem_resampling": p.dem_resampling,
                "rem": {
                    "enabled": p.rem.enabled,
                    "sampling_interval_pixels": p.rem.sampling_interval_pixels,
                    "idw_power": p.rem.idw_power,
                },
            },
            "outputs": {
                "directory": str(o.directory),
                "intermediates": o.intermediates,
                "final": o.final,
                "save_conditioned_dem": o.save_conditioned_dem,
                "save_flow_direction": o.save_flow_direction,
                "save_flow_accumulation": o.save_flow_accumulation,
                "save_drainage_network": o.save_drainage_network,
                "save_hand": o.save_hand,
                "rem": o.rem,
                "save_basins": o.save_basins,
                "save_rem_samples": o.save_rem_samples,
                "save_rem_drainage_surface": o.save_rem_drainage_surface,
            },
            "execution": {
                "overwrite": self.execution.overwrite,
                "verbose": self.execution.verbose,
                "whitebox_executable": (
                    str(self.execution.whitebox_executable)
                    if self.execution.whitebox_executable
                    else None
                ),
                "resume": self.execution.resume,
                "max_procs": self.execution.max_procs,
                "heartbeat_seconds": self.execution.heartbeat_seconds,
                "whitebox_version": self.execution.whitebox_version,
                "scratch_directory": (
                    str(self.execution.scratch_directory) if self.execution.scratch_directory else None
                ),
            },
        }


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
_MISSING = object()


def _section(
    data: Any,
    name: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> Mapping[str, Any]:
    """Validate that ``data`` is a mapping with known keys and all required keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Configuration section '{name}' must be a mapping (key: value pairs), "
            f"got {type(data).__name__}."
        )
    required, optional = tuple(required), tuple(optional)
    unknown = sorted(set(data) - set(required) - set(optional))
    if unknown:
        allowed = ", ".join(sorted(set(required) | set(optional)))
        raise ConfigError(
            f"Unknown parameter(s) in '{name}': {', '.join(map(str, unknown))}. "
            f"Allowed parameters: {allowed}."
        )
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigError(
            f"Missing required parameter(s) in '{name}': {', '.join(missing)}. "
            "Scientific parameters have no defaults and must be set explicitly."
        )
    return data


def _string(value: Any, name: str, *, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string, got {value!r}.")
    return value.strip()


def _choice(value: Any, name: str, choices: Iterable[str]) -> str:
    choices = tuple(choices)
    if not isinstance(value, str) or value.strip().lower() not in choices:
        raise ConfigError(f"'{name}' must be one of {list(choices)}, got {value!r}.")
    return value.strip().lower()


def _number(value: Any, name: str, *, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"'{name}' must be finite, got {value!r}.")
    if (value < minimum) if inclusive else (value <= minimum):
        rel = ">=" if inclusive else ">"
        raise ConfigError(f"'{name}' must be {rel} {minimum:g}, got {value:g}.")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{name}' must be true or false (YAML boolean), got {value!r}.")
    return value


def _path(value: Any, name: str) -> Path:
    text = _string(value, name)
    return Path(text).expanduser().resolve()


def _subdir(value: Any, name: str) -> str:
    text = _string(value, name)
    pure = PurePath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ConfigError(
            f"'{name}' must be a relative directory name inside outputs.directory "
            f"(no absolute paths or '..'), got {text!r}."
        )
    return text


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_config(data: Any, *, source_path: Optional[Path] = None, source_sha256: Optional[str] = None) -> Config:
    """Validate a configuration mapping and return a :class:`Config`."""
    top = _section(data, "<root>", ("inputs", "processing", "outputs"), ("project", "execution"))

    project = _section(top.get("project", {}), "project", (), ("name",))
    project_name = _string(project.get("name", DEFAULT_PROJECT_NAME), "project.name")

    inp = _section(top["inputs"], "inputs", ("dem", "pilot_area", "vertical_units"), ("pilot_layer",))
    inputs = InputsConfig(
        dem=_path(inp["dem"], "inputs.dem"),
        pilot_area=_path(inp["pilot_area"], "inputs.pilot_area") if inp["pilot_area"] is not None else None,
        pilot_layer=_string(inp.get("pilot_layer"), "inputs.pilot_layer", allow_none=True),
        vertical_units=_choice(inp["vertical_units"], "inputs.vertical_units", VERTICAL_UNITS),
    )

    proc = _section(
        top["processing"],
        "processing",
        ("buffer_distance", "conditioning", "flow", "drainage", "hand"),
        ("buffer_units", "crs", "target_resolution", "dem_resampling", "rem"),
    )
    cond = _section(
        proc["conditioning"], "processing.conditioning", ("method",), ("breach_least_cost", "fill_remaining")
    )
    cond_method = _choice(cond["method"], "processing.conditioning.method", CONDITIONING_METHODS)
    breach_lc, fill_remaining = None, False
    if cond_method == "breach_least_cost":
        if "breach_least_cost" not in cond:
            raise ConfigError(
                "'processing.conditioning.breach_least_cost' is required when the method is 'breach_least_cost' "
                "(at least 'max_distance_cells', the maximum breach path length in cells)."
            )
        bl = _section(
            cond["breach_least_cost"], "processing.conditioning.breach_least_cost", ("max_distance_cells",),
            ("max_cost", "minimize_distance", "flat_increment"),
        )
        dist = _number(bl["max_distance_cells"], "processing.conditioning.breach_least_cost.max_distance_cells",
                       minimum=1.0, inclusive=True)
        if isinstance(bl["max_distance_cells"], bool) or dist != int(dist):
            raise ConfigError(
                "'processing.conditioning.breach_least_cost.max_distance_cells' must be a positive whole number "
                f"of cells, got {bl['max_distance_cells']!r}."
            )
        max_cost = bl.get("max_cost")
        if max_cost is not None:
            max_cost = float(_number(max_cost, "processing.conditioning.breach_least_cost.max_cost", minimum=0.0, inclusive=False))
        flat_inc = bl.get("flat_increment")
        if flat_inc is not None:
            flat_inc = float(_number(flat_inc, "processing.conditioning.breach_least_cost.flat_increment", minimum=0.0, inclusive=False))
        breach_lc = BreachLeastCostConfig(
            max_distance_cells=int(dist), max_cost=max_cost,
            minimize_distance=_boolean(bl.get("minimize_distance", False), "processing.conditioning.breach_least_cost.minimize_distance"),
            flat_increment=flat_inc,
        )
        fill_remaining = _boolean(cond.get("fill_remaining", False), "processing.conditioning.fill_remaining")
    elif "breach_least_cost" in cond or "fill_remaining" in cond:
        raise ConfigError(
            "'processing.conditioning.breach_least_cost' and 'processing.conditioning.fill_remaining' only apply "
            f"to the method 'breach_least_cost' (configured method: '{cond_method}'); remove them or change the method."
        )
    flow = _section(proc["flow"], "processing.flow", ("algorithm",))
    drain = _section(proc["drainage"], "processing.drainage", ("threshold_type", "threshold_value"))
    hand = _section(proc["hand"], "processing.hand", ("threshold",))

    threshold_type = _choice(drain["threshold_type"], "processing.drainage.threshold_type", DRAINAGE_THRESHOLD_TYPES)
    threshold_value = _number(drain["threshold_value"], "processing.drainage.threshold_value", minimum=0.0, inclusive=False)
    if threshold_type == "cells" and threshold_value != int(threshold_value):
        raise ConfigError(
            "'processing.drainage.threshold_value' must be a whole number of cells when "
            f"threshold_type is 'cells', got {threshold_value:g}."
        )

    crs_spec = _string(proc.get("crs", DEFAULT_PROCESSING_CRS), "processing.crs")
    try:
        resolve_processing_crs(crs_spec)  # fail early: must exist in the installed PROJ and be projected
    except InputError as exc:
        raise ConfigError(str(exc)) from exc
    target_resolution = proc.get("target_resolution")
    if target_resolution is not None:
        target_resolution = _number(target_resolution, "processing.target_resolution", minimum=0.0, inclusive=False)
    resampling = _choice(proc.get("dem_resampling", "bilinear"), "processing.dem_resampling", DEM_RESAMPLING_METHODS)

    rem_raw = _section(proc.get("rem", {}), "processing.rem", (), ("enabled", "sampling_interval_pixels", "idw_power"))
    rem_enabled = _boolean(rem_raw.get("enabled", False), "processing.rem.enabled")
    interval = rem_raw.get("sampling_interval_pixels")
    if interval is not None:
        interval_f = _number(interval, "processing.rem.sampling_interval_pixels", minimum=1.0, inclusive=True)
        if isinstance(interval, bool) or interval_f != int(interval_f):
            raise ConfigError(
                f"'processing.rem.sampling_interval_pixels' must be a positive whole number of pixels, got {interval!r}."
            )
        interval = int(interval_f)
    elif rem_enabled:
        raise ConfigError(
            "'processing.rem.sampling_interval_pixels' is required when 'processing.rem.enabled' is true "
            "(positive integer; smaller values give more drainage samples)."
        )
    idw_power = _number(rem_raw.get("idw_power", DEFAULT_IDW_POWER), "processing.rem.idw_power", minimum=0.0, inclusive=False)
    if not math.isfinite(idw_power):
        raise ConfigError("'processing.rem.idw_power' must be finite.")
    rem = RemConfig(enabled=rem_enabled, sampling_interval_pixels=interval, idw_power=float(idw_power))

    processing = ProcessingConfig(
        buffer_distance=_number(proc["buffer_distance"], "processing.buffer_distance", minimum=0.0, inclusive=True),
        buffer_units=_choice(proc.get("buffer_units", "meters"), "processing.buffer_units", BUFFER_UNITS),
        conditioning_method=cond_method,
        flow_algorithm=_choice(flow["algorithm"], "processing.flow.algorithm", FLOW_ALGORITHMS),
        drainage_threshold_type=threshold_type,
        drainage_threshold_value=threshold_value,
        hand_threshold=_number(hand["threshold"], "processing.hand.threshold", minimum=0.0, inclusive=True),
        crs=crs_spec,
        target_resolution=target_resolution,
        dem_resampling=resampling,
        rem=rem,
        breach_least_cost=breach_lc,
        fill_remaining=fill_remaining,
    )

    out = _section(
        top["outputs"],
        "outputs",
        ("directory",),
        (
            "intermediates",
            "final",
            "save_conditioned_dem",
            "save_flow_direction",
            "save_flow_accumulation",
            "save_drainage_network",
            "save_hand",
            "rem",
            "save_basins",
            "save_rem_samples",
            "save_rem_drainage_surface",
        ),
    )
    intermediates = _subdir(out.get("intermediates", "intermediate"), "outputs.intermediates")
    final = _subdir(out.get("final", "final"), "outputs.final")
    rem_dir = _subdir(out.get("rem", "rem"), "outputs.rem")
    if len({intermediates, final, rem_dir}) != 3:
        raise ConfigError("'outputs.intermediates', 'outputs.final' and 'outputs.rem' must be different directories.")
    outputs = OutputsConfig(
        directory=_path(out["directory"], "outputs.directory"),
        intermediates=intermediates,
        final=final,
        save_conditioned_dem=_boolean(out.get("save_conditioned_dem", True), "outputs.save_conditioned_dem"),
        save_flow_direction=_boolean(out.get("save_flow_direction", True), "outputs.save_flow_direction"),
        save_flow_accumulation=_boolean(out.get("save_flow_accumulation", True), "outputs.save_flow_accumulation"),
        save_drainage_network=_boolean(out.get("save_drainage_network", True), "outputs.save_drainage_network"),
        save_hand=_boolean(out.get("save_hand", True), "outputs.save_hand"),
        rem=rem_dir,
        save_basins=_boolean(out.get("save_basins", True), "outputs.save_basins"),
        save_rem_samples=_boolean(out.get("save_rem_samples", True), "outputs.save_rem_samples"),
        save_rem_drainage_surface=_boolean(out.get("save_rem_drainage_surface", True), "outputs.save_rem_drainage_surface"),
    )

    ex = _section(
        top.get("execution", {}),
        "execution",
        (),
        (
            "overwrite", "verbose", "whitebox_executable", "resume", "max_procs",
            "heartbeat_seconds", "whitebox_version", "scratch_directory",
        ),
    )
    wbt_exe = ex.get("whitebox_executable")
    scratch = ex.get("scratch_directory")
    max_procs = ex.get("max_procs")
    if max_procs is not None:
        max_procs = _number(max_procs, "execution.max_procs", minimum=1.0, inclusive=True)
        if max_procs != int(max_procs):
            raise ConfigError(f"'execution.max_procs' must be a whole number, got {max_procs:g}.")
        max_procs = int(max_procs)
    wbt_version = _string(ex.get("whitebox_version"), "execution.whitebox_version", allow_none=True)
    execution = ExecutionConfig(
        overwrite=_boolean(ex.get("overwrite", False), "execution.overwrite"),
        verbose=_boolean(ex.get("verbose", True), "execution.verbose"),
        whitebox_executable=_path(wbt_exe, "execution.whitebox_executable") if wbt_exe is not None else None,
        resume=_boolean(ex.get("resume", False), "execution.resume"),
        max_procs=max_procs,
        heartbeat_seconds=_number(ex.get("heartbeat_seconds", 300), "execution.heartbeat_seconds", minimum=1.0, inclusive=True),
        whitebox_version=wbt_version,
        scratch_directory=_path(scratch, "execution.scratch_directory") if scratch is not None else None,
    )
    if execution.overwrite and execution.resume:
        raise ConfigError(
            "'execution.overwrite' and 'execution.resume' cannot both be true: overwrite restarts from "
            "scratch, resume reuses the up-to-date results of a previous run. Choose one."
        )

    return Config(
        project_name=project_name,
        inputs=inputs,
        processing=processing,
        outputs=outputs,
        execution=execution,
        source_path=source_path,
        source_sha256=source_sha256,
        raw=dict(data),
    )


def load_config(path: Path | str) -> Config:
    """Read and validate a YAML configuration file."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        raw_bytes = path.read_bytes()
        data = yaml.safe_load(raw_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML syntax in {path}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"Configuration file {path} is empty.")
    return parse_config(
        data,
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
