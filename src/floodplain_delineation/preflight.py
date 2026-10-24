"""Pre-run resource estimate: logs what the run will need and warns before heavy work starts.

The numbers are estimates meant to catch obvious problems (a full disk, a container memory
limit far below the raster size) *before* a multi-hour run. The real peaks are measured and
logged per tool afterwards (see ``performance`` in the processing summary).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import resources
from .config import Config
from .logging_utils import WarningCollector, get_logger
from .outputs import OutputPlan

log = get_logger("preflight")

#: Conservative on-disk bytes per cell: extent 4 + conditioned 4 + direction 4 + accumulation 4
#: + drainage 1 + HAND 4 + final <=1, plus one raw WhiteboxTools output (up to 8) while it is
#: being rewritten.
DISK_BYTES_PER_CELL_KEPT = 22
DISK_BYTES_PER_CELL_RAW = 8


def run_preflight(
    width: int, height: int, plan: OutputPlan, config: Config, warnings: WarningCollector
) -> Dict[str, Any]:
    n_cells = int(width) * int(height)
    raster_f32 = n_cells * 4
    raster_f64 = n_cells * 8
    kept = n_cells * DISK_BYTES_PER_CELL_KEPT
    raw = n_cells * DISK_BYTES_PER_CELL_RAW
    scratch = config.execution.scratch_directory

    free_out = resources.free_disk_bytes(plan.root)
    free_scratch = resources.free_disk_bytes(scratch) if scratch is not None else None
    required_out = kept + (0 if scratch is not None else raw)
    mem = resources.available_memory_bytes()
    cpus = resources.cpu_count()

    info: Dict[str, Any] = {
        "processing_window_cells": n_cells,
        "one_raster_float32_gib": resources.gib(raster_f32),
        "one_raster_float64_gib": resources.gib(raster_f64),
        "disk_required_outputs_gib": resources.gib(required_out),
        "disk_free_outputs_gib": resources.gib(free_out),
        "disk_required_scratch_gib": resources.gib(raw) if scratch is not None else None,
        "disk_free_scratch_gib": resources.gib(free_scratch),
        "memory_available_gib": resources.gib(mem["machine_available"]),
        "memory_cgroup_limit_gib": resources.gib(mem["cgroup_limit"]),
        "cpus_available": cpus,
        "max_procs_requested": config.execution.max_procs,
        "note": "estimates; measured peaks are logged after each WhiteboxTools run",
    }
    log.info(
        "Processing window: %d cells; one raster = %s (float32) / %s (float64).",
        n_cells, resources.format_bytes(raster_f32), resources.format_bytes(raster_f64),
    )
    log.info(
        "Disk: ~%s needed in %s (free: %s)%s.",
        resources.format_bytes(required_out), plan.root, resources.format_bytes(free_out),
        f"; ~{resources.format_bytes(raw)} in scratch {scratch} (free: {resources.format_bytes(free_scratch)})"
        if scratch is not None else "",
    )
    log.info(
        "Memory: machine available %s, container limit %s; CPUs available: %d%s.",
        resources.format_bytes(mem["machine_available"]),
        resources.format_bytes(mem["cgroup_limit"]) if mem["cgroup_limit"] else "none detected",
        cpus,
        f", max_procs={config.execution.max_procs}" if config.execution.max_procs else "",
    )
    if free_out is not None and free_out < required_out:
        warnings.add(
            f"Free disk space in {plan.root} ({resources.format_bytes(free_out)}) is below the conservative "
            f"estimate for this run ({resources.format_bytes(required_out)}). Free space, choose another "
            "outputs.directory, or set execution.save_* options to false only after checking the estimate."
        )
    if scratch is not None and free_scratch is not None and free_scratch < raw:
        warnings.add(
            f"Free disk space in the scratch directory {scratch} ({resources.format_bytes(free_scratch)}) is "
            f"below the estimate for one raw WhiteboxTools output ({resources.format_bytes(raw)})."
        )
    eff: Optional[int] = mem["effective"]
    if eff is not None and eff < 4 * raster_f64:
        warnings.add(
            f"Available memory ({resources.format_bytes(eff)}) is less than four float64 rasters of this size "
            f"({resources.format_bytes(4 * raster_f64)}). WhiteboxTools loads whole rasters in memory; the run "
            "may be killed by the operating system (heuristic check)."
        )
    if config.execution.max_procs is not None and config.execution.max_procs > cpus:
        warnings.add(
            f"execution.max_procs ({config.execution.max_procs}) exceeds the CPUs available to this process ({cpus})."
        )
    return info
