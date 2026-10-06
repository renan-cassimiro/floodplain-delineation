"""High-level orchestration of the workflow (Stages 1-7 plus metadata)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from .config import Config
from .delineation import PRODUCT_TITLE, build_final_mask
from .exceptions import FloodplainError
from .hand import compute_hand
from .hydrology import (
    compute_flow_accumulation,
    compute_flow_direction,
    drainage_diagnostics,
    extract_drainage,
    resolve_drainage_threshold,
)
from .inputs import prepare_spatial_domain, read_dem_info, validate_input_files
from .logging_utils import WarningCollector, get_logger, setup_logging
from .outputs import (
    cleanup,
    collect_versions,
    plan_outputs,
    prepare_output_dirs,
    write_processing_area,
    write_summary,
)
from .preprocessing import CONDITIONING_TOOLS, condition_dem, extract_processing_dem
from .rasters import read_band
from .spatial import cell_geometry
from .wbt import WhiteboxRunner

log = get_logger("pipeline")

LIMITATIONS = [
    "The output is a preliminary topographic approximation of potentially inundatable terrain; it is "
    "NOT observed inundation, a hydraulic simulation, or a validated floodplain boundary.",
    "No validation or accuracy assessment against reference data was performed.",
    "Results depend on DEM vertical accuracy and artefacts (vegetation/building bias, voids, striping).",
    "Results are sensitive to the drainage threshold; the derived drainage network controls HAND.",
    "The HAND threshold is a user-chosen elevation above drainage, not a measured water level or flood return period.",
    "A processing buffer cannot guarantee correct upstream hydrology if the DEM does not extend "
    "sufficiently beyond the area of interest; edge effects may remain near the extent boundary.",
    "Single-direction D8 routing is used; flow over very flat terrain depends on WhiteboxTools flat handling.",
]


def run_pipeline(config: Config) -> Dict[str, Any]:
    """Run the complete workflow and return the processing summary."""
    started = datetime.now(timezone.utc)

    # --- checks that must pass before anything is written --------------------
    validate_input_files(config)
    plan = plan_outputs(config)
    prepare_output_dirs(plan, config)
    setup_logging(plan.paths["processing_log"], config.execution.verbose)
    warnings = WarningCollector()

    summary: Dict[str, Any] = {
        "status": "running",
        "product": PRODUCT_TITLE,
        "started_utc": started.isoformat(),
        "configuration": {
            "effective": config.to_dict(),
            "source_file": str(config.source_path) if config.source_path else None,
            "source_sha256": config.source_sha256,
        },
        "inputs": {"dem": str(config.inputs.dem), "pilot_area": str(config.inputs.pilot_area)},
        "stages": {},
    }
    p = config.processing

    try:
        log.info("=== %s ===", config.project_name)

        # --- tool availability (fail fast, before heavy work) ----------------
        runner = WhiteboxRunner(config.execution.whitebox_executable)
        wbt_version = runner.version()
        tool_name = CONDITIONING_TOOLS[p.conditioning_method][0]
        runner.verify_tools([tool_name, "D8Pointer", "D8FlowAccumulation", "ExtractStreams", "ElevationAboveStream"])
        log.info("Using %s", wbt_version)

        # --- Stage 1 ---------------------------------------------------------
        log.info("Stage 1: input validation and spatial preparation")
        dem = read_dem_info(config.inputs.dem, warnings)
        domain = prepare_spatial_domain(dem, config, warnings)
        cell = cell_geometry(dem.pcrs, domain.window_transform, domain.extent_bounds)
        write_processing_area(plan.paths["processing_area"], domain, dem.crs)
        summary["spatial"] = {
            "dem_crs": dem.crs.to_string(),
            "pilot_source_crs": domain.source_crs,
            "output_crs": dem.crs.to_string(),
            "reprojection_note": "Only the pilot geometry is reprojected (to the DEM CRS); the DEM is never reprojected or resampled.",
            "dem_resolution_crs_units": list(dem.resolution),
            "cell_size_m": [cell.dx_m, cell.dy_m],
            "cell_area_km2": cell.area_km2,
            "vertical_units": config.inputs.vertical_units,
            "dem_nodata": dem.nodata,
            "dem_dtype": dem.dtype,
            "pilot_features": domain.n_features,
            "pilot_coverage_by_dem": domain.coverage_fraction,
            "processing_extent_bounds": list(domain.extent_bounds),
            "requested_extent_bounds": list(domain.requested_bounds),
            "processing_window": [int(domain.window.col_off), int(domain.window.row_off),
                                  int(domain.window.width), int(domain.window.height)],
            "buffer": domain.buffer_info,
            "buffer_truncated_sides": domain.truncated_sides,
        }

        # --- Stage 2 ---------------------------------------------------------
        log.info("Stage 2: DEM extraction and hydrological conditioning")
        extent_info = extract_processing_dem(dem, domain, plan.paths["dem_processing_extent"], warnings)
        cond_diag, dem_valid, grid = condition_dem(
            runner, p.conditioning_method, plan.paths["dem_processing_extent"],
            plan.paths["conditioned_dem"], plan.work_dir, warnings,
        )
        summary["stages"]["dem_extraction"] = extent_info
        summary["stages"]["conditioning"] = {"method": p.conditioning_method, **cond_diag}

        # --- Stage 3 ---------------------------------------------------------
        log.info("Stage 3: D8 flow direction and flow accumulation")
        fdir = compute_flow_direction(runner, plan.paths["conditioned_dem"], plan.paths["flow_direction"],
                                      plan.work_dir, dem_valid, grid, warnings)
        facc = compute_flow_accumulation(runner, plan.paths["flow_direction"], plan.paths["flow_accumulation"],
                                         plan.work_dir, dem_valid, grid, warnings)
        summary["stages"]["flow_routing"] = {"algorithm": p.flow_algorithm, "direction": fdir, "accumulation": facc}

        # --- Stage 4 ---------------------------------------------------------
        log.info("Stage 4: drainage network extraction")
        thr = resolve_drainage_threshold(p.drainage_threshold_type, p.drainage_threshold_value, cell, warnings)
        log.info("Drainage threshold: %s %g -> %d contributing cells (%.4f km2)",
                 thr["type"], thr["configured_value"], thr["effective_cells"], thr["effective_area_km2"])
        drain = extract_drainage(
            runner, plan.paths["flow_accumulation"], plan.paths["drainage_network"],
            plan.work_dir / "drainage_raw.tif", plan.work_dir, thr["effective_cells"], dem_valid, grid, warnings,
        )
        diag = drainage_diagnostics(drain, plan.paths["flow_direction"], cell)
        log.info("Drainage: %d cells (%.2f%% of the processing area), ~%.1f km.",
                 diag["drainage_cells"], diag["percent_of_processing_area"], diag["approx_length_km"])
        summary["stages"]["drainage"] = {"threshold": thr, "diagnostics": diag}

        # --- Stage 5 ---------------------------------------------------------
        log.info("Stage 5: HAND")
        hand_stats = compute_hand(runner, plan.paths["conditioned_dem"], plan.paths["drainage_network"],
                                  plan.paths["hand"], plan.work_dir, dem_valid, drain, grid, warnings)
        summary["stages"]["hand"] = hand_stats

        # --- Stages 6-7 ------------------------------------------------------
        log.info("Stages 6-7: HAND threshold (%g %s), clipping and export", p.hand_threshold, config.inputs.vertical_units)
        final = build_final_mask(plan.paths["hand"], domain.pilot, p.hand_threshold,
                                 plan.paths["potential_floodplain"], config.inputs.vertical_units, cell)
        summary["stages"]["floodplain"] = {
            "hand_threshold": p.hand_threshold, "hand_threshold_units": config.inputs.vertical_units, **final,
        }

        # --- wrap-up ---------------------------------------------------------
        removed = cleanup(plan)
        summary["outputs"] = {
            k: (str(v) if v.exists() else None) for k, v in plan.paths.items() if k != "processing_summary"
        }
        summary["outputs"]["processing_summary"] = str(plan.paths["processing_summary"])
        summary["intermediates_not_saved"] = removed
        summary["status"] = "success"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        log.error("Processing failed: %s", exc)
        raise
    finally:
        finished = datetime.now(timezone.utc)
        summary["finished_utc"] = finished.isoformat()
        summary["duration_seconds"] = (finished - started).total_seconds()
        try:
            summary["software_versions"] = collect_versions(locals().get("wbt_version"))
            if "runner" in locals():
                summary["whitebox_calls"] = runner.calls
        except Exception:  # pragma: no cover - never mask the primary error
            pass
        summary["warnings"] = warnings.messages
        summary["limitations"] = LIMITATIONS
        write_summary(plan.paths["processing_summary"], summary)

    log.info("Done. Final product: %s", plan.paths["potential_floodplain"])
    return summary
