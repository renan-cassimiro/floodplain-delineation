"""Output planning, overwrite handling, software versions and the processing summary."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from . import __version__
from .config import Config
from .exceptions import OutputError
from .logging_utils import get_logger

log = get_logger("outputs")

REQUIRED_NAMES = {
    "conditioned_dem": "conditioned_dem.tif",
    "flow_direction": "flow_direction.tif",
    "flow_accumulation": "flow_accumulation.tif",
    "drainage_network": "drainage_network.tif",
    "hand": "hand.tif",
}
FINAL_NAME = "potential_floodplain.tif"
EXTENT_NAME = "dem_processing_extent.tif"


@dataclass(frozen=True)
class OutputPlan:
    root: Path
    intermediate_dir: Path
    final_dir: Path
    work_dir: Path
    paths: Dict[str, Path]
    save_flags: Dict[str, bool]

    @property
    def all_files(self) -> List[Path]:
        return list(self.paths.values())


def plan_outputs(config: Config) -> OutputPlan:
    o = config.outputs
    inter = o.directory / o.intermediates
    final = o.directory / o.final
    paths = {"dem_processing_extent": inter / EXTENT_NAME}
    paths.update({k: inter / v for k, v in REQUIRED_NAMES.items()})
    paths["potential_floodplain"] = final / FINAL_NAME
    paths["processing_summary"] = o.directory / "processing_summary.json"
    paths["processing_log"] = o.directory / "processing.log"
    paths["processing_area"] = o.directory / "processing_area.gpkg"
    flags = {
        "conditioned_dem": o.save_conditioned_dem,
        "flow_direction": o.save_flow_direction,
        "flow_accumulation": o.save_flow_accumulation,
        "drainage_network": o.save_drainage_network,
        "hand": o.save_hand,
    }
    return OutputPlan(o.directory, inter, final, inter / ".wbt_tmp", paths, flags)


def prepare_output_dirs(plan: OutputPlan, config: Config) -> None:
    """Check conflicts (inputs / existing files) BEFORE any processing, then create directories."""
    inputs = {config.inputs.dem.resolve(), config.inputs.pilot_area.resolve()}
    clash = [p for p in plan.all_files if p.resolve() in inputs]
    if clash:
        raise OutputError(f"Output path(s) coincide with input files: {clash}. Change outputs.directory.")
    existing = [p for p in plan.all_files if p.exists()]
    if existing and not config.execution.overwrite:
        names = ", ".join(str(p) for p in existing[:6]) + (" ..." if len(existing) > 6 else "")
        raise OutputError(
            f"Output file(s) already exist: {names}. Set execution.overwrite: true or choose another "
            "outputs.directory."
        )
    try:
        for d in (plan.intermediate_dir, plan.final_dir):
            d.mkdir(parents=True, exist_ok=True)
        probe = plan.root / ".write_test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise OutputError(f"Output directory {plan.root} is not writable: {exc}") from exc
    for p in existing:  # only reached when overwrite is true
        try:
            p.unlink()
        except OSError as exc:
            raise OutputError(f"Cannot overwrite {p}: {exc}") from exc
    plan.work_dir.mkdir(parents=True, exist_ok=True)


def cleanup(plan: OutputPlan) -> List[str]:
    """Remove the temporary WhiteboxTools directory and intermediates the user chose not to keep."""
    removed = []
    for key, keep in plan.save_flags.items():
        if not keep and plan.paths[key].exists():
            plan.paths[key].unlink()
            removed.append(key)
    try:
        for f in plan.work_dir.glob("*"):
            f.unlink()
        plan.work_dir.rmdir()
    except OSError:
        pass
    return removed


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect_versions(wbt_version: Optional[str]) -> Dict[str, str]:
    import rasterio

    return {
        "floodplain_delineation": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "gdal": getattr(rasterio, "__gdal_version__", "unknown"),
        "geopandas": gpd.__version__,
        "shapely": _version("shapely"),
        "pyproj": _version("pyproj"),
        "pyyaml": _version("PyYAML"),
        "whitebox_python_package": _version("whitebox"),
        "whitebox_tools_executable": wbt_version or "unknown",
    }


def write_processing_area(path: Path, domain, dem_crs) -> None:
    """GeoPackage with the pilot area (clipped to the DEM), the full pilot area and the processing extent."""
    layers = {
        "pilot_area_in_dem": domain.pilot,
        "pilot_area_full": domain.pilot_full,
        "processing_extent": box(*domain.extent_bounds),
        "requested_extent": box(*domain.requested_bounds),
    }
    for i, (name, geom) in enumerate(layers.items()):
        gpd.GeoDataFrame({"name": [name]}, geometry=[geom], crs=dem_crs.to_wkt()).to_file(
            path, layer=name, driver="GPKG", mode="w" if i == 0 else "a"
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    except OSError as exc:
        raise OutputError(f"Cannot write processing summary {path}: {exc}") from exc
