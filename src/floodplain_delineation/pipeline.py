"""High-level orchestration of the workflow (Stages 1-7 plus metadata).

Stages 2-5 (DEM extraction, conditioning, flow direction, flow accumulation, drainage, HAND) are
resumable: see :mod:`stages` and :mod:`manifest`. Stage 1 (inputs/spatial preparation) and the final
HAND-threshold stage are cheap and always run.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict

import rasterio

from . import __version__, resources
from .config import Config
from .delineation import PRODUCT_TITLE, build_final_mask
from .exceptions import FloodplainError, WhiteboxError
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
from .manifest import StageManifest, file_signature, fingerprint
from .outputs import (
    cleanup,
    collect_versions,
    plan_outputs,
    prepare_output_dirs,
    write_processing_area,
    write_summary,
)
from .preflight import run_preflight
from .preprocessing import CONDITIONING_TOOLS, condition_dem, extract_processing_dem
from .rasters import GridInfo
from .spatial import cell_geometry
from .stages import StageExecutor, StageSpec
from .wbt import TESTED_WBT_VERSIONS, WhiteboxRunner, parse_version

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


def _check_wbt_version(pin: Any, version_text: str, warnings: WarningCollector) -> None:
    """Enforce ``execution.whitebox_version`` when set; otherwise warn about untested versions."""
    found = parse_version(version_text)
    if pin is not None:
        if found != pin:
            raise WhiteboxError(
                f"execution.whitebox_version is '{pin}' but the installed WhiteboxTools reports "
                f"'{version_text}'. Install that version or update the configuration."
            )
        return
    if found not in TESTED_WBT_VERSIONS:
        warnings.add(
            f"WhiteboxTools version '{found or version_text}' is not covered by this project's automated tests "
            f"(tested: {', '.join(TESTED_WBT_VERSIONS)}). Every stage output is still validated, and tool rasters "
            "are rewritten with GDAL before reuse. Set execution.whitebox_version to pin the version you validated."
        )


#: GDAL settings for the whole run. Rasters are read/written sequentially in stripes, so a large
#: block cache is useless (GDAL's default is 5 % of the RAM, i.e. ~5 GB on a 96 GB machine);
#: compression threads speed up writing the final DEFLATE-compressed map.
GDAL_OPTIONS = {"GDAL_CACHEMAX": 256, "GDAL_NUM_THREADS": "ALL_CPUS"}


def run_pipeline(config: Config) -> Dict[str, Any]:
    """Run the complete workflow and return the processing summary."""
    with rasterio.Env(**GDAL_OPTIONS):
        return _run_pipeline(config)


def _run_pipeline(config: Config) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()

    # --- checks that must pass before anything is written --------------------
    validate_input_files(config)
    plan = plan_outputs(config)
    prepare_output_dirs(plan, config)
    setup_logging(plan.paths["processing_log"], config.execution.verbose, append=config.execution.resume)
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
        "inputs": {
            "dem": str(config.inputs.dem),
            "pilot_area": str(config.inputs.pilot_area) if config.inputs.pilot_area else None,
        },
        "stages": {},
    }
    p = config.processing
    executor = None  # defined once the manifest exists

    try:
        log.info("=== %s ===", config.project_name)
        if config.execution.resume:
            log.info("Resume mode: up-to-date stages of the previous run will be reused.")

        # --- tool availability (fail fast, before heavy work) ----------------
        runner = WhiteboxRunner(
            config.execution.whitebox_executable,
            max_procs=config.execution.max_procs,
            heartbeat_seconds=config.execution.heartbeat_seconds,
        )
        wbt_version = runner.version()
        tool_name = CONDITIONING_TOOLS[p.conditioning_method][0]
        runner.verify_tools([tool_name, "D8Pointer", "D8FlowAccumulation", "ExtractStreams", "ElevationAboveStream"])
        log.info("Using %s", wbt_version)
        _check_wbt_version(config.execution.whitebox_version, wbt_version, warnings)

        # --- Stage 1 ---------------------------------------------------------
        log.info("Stage 1: input validation and spatial preparation")
        dem = read_dem_info(config.inputs.dem, warnings)
        domain = prepare_spatial_domain(dem, config, warnings)
        cell = cell_geometry(dem.pcrs, domain.window_transform, domain.extent_bounds)
        write_processing_area(plan.paths["processing_area"], domain, dem.crs)
        win_w, win_h = int(domain.window.width), int(domain.window.height)
        grid = GridInfo(win_h, win_w, domain.window_transform, dem.crs)
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
            "processing_window": [int(domain.window.col_off), int(domain.window.row_off), win_w, win_h],
            "buffer": domain.buffer_info,
            "buffer_truncated_sides": domain.truncated_sides,
        }
        summary["preflight"] = run_preflight(win_w, win_h, plan, config, warnings)

        thr = resolve_drainage_threshold(p.drainage_threshold_type, p.drainage_threshold_value, cell, warnings)
        log.info("Drainage threshold: %s %g -> %d contributing cells (%.4f km2)",
                 thr["type"], thr["configured_value"], thr["effective_cells"], thr["effective_area_km2"])

        # --- stage graph (Stages 2-5) -----------------------------------------
        paths, work = plan.paths, plan.work_dir
        pilot_sig = (
            {**file_signature(config.inputs.pilot_area), "layer": config.inputs.pilot_layer}
            if config.inputs.pilot_area else None
        )
        wbt_id = parse_version(wbt_version) or wbt_version
        common = {"package": __version__}
        fp_ext = fingerprint(
            "dem_extraction",
            {**common, "dem": file_signature(config.inputs.dem), "window": [int(domain.window.col_off),
             int(domain.window.row_off), win_w, win_h]},
        )
        fp_cond = fingerprint("conditioning", {**common, "method": p.conditioning_method, "tool": tool_name, "wbt": wbt_id}, [fp_ext])
        fp_dir = fingerprint("flow_direction", {**common, "wbt": wbt_id}, [fp_cond])
        fp_acc = fingerprint("flow_accumulation", {**common, "wbt": wbt_id}, [fp_dir])
        fp_drain = fingerprint("drainage", {**common, "wbt": wbt_id, "threshold": thr}, [fp_acc, fp_dir])
        fp_hand = fingerprint("hand", {**common, "wbt": wbt_id}, [fp_cond, fp_drain])

        manifest = StageManifest(paths["stage_manifest"])
        if config.execution.resume:
            manifest.load()
        executor = StageExecutor(manifest, warnings, config.execution.resume, tool_peak=lambda: runner.peak_rss_bytes)

        def run_extract() -> Dict[str, Any]:
            info = extract_processing_dem(dem, domain, paths["dem_processing_extent"], warnings)
            info["pilot_signature"] = pilot_sig
            return info

        def run_conditioning() -> Dict[str, Any]:
            return condition_dem(runner, p.conditioning_method, paths["dem_processing_extent"],
                                 paths["conditioned_dem"], work, warnings)

        def run_direction() -> Dict[str, Any]:
            return compute_flow_direction(runner, paths["conditioned_dem"], paths["flow_direction"], work, grid, warnings)

        def run_accumulation() -> Dict[str, Any]:
            return compute_flow_accumulation(runner, paths["flow_direction"], paths["flow_accumulation"], work,
                                             paths["conditioned_dem"], grid, warnings)

        def run_drainage() -> Dict[str, Any]:
            extract_drainage(runner, paths["flow_accumulation"], paths["drainage_network"], work / "drainage_raw.tif",
                             work, thr["effective_cells"], paths["conditioned_dem"], grid, warnings)
            diag = drainage_diagnostics(paths["drainage_network"], paths["flow_direction"], cell)
            log.info("Drainage: %d cells (%.2f%% of the processing area), ~%.1f km.",
                     diag["drainage_cells"], diag["percent_of_processing_area"], diag["approx_length_km"])
            return {"threshold": thr, "diagnostics": diag}

        def run_hand() -> Dict[str, Any]:
            return compute_hand(runner, paths["conditioned_dem"], paths["drainage_network"], paths["hand"],
                                work, grid, warnings)

        for spec in (
            StageSpec("dem_extraction", "DEM extraction", [paths["dem_processing_extent"]], fp_ext, [], run_extract),
            StageSpec("conditioning", f"hydrological conditioning ({p.conditioning_method})",
                      [paths["conditioned_dem"]], fp_cond, ["dem_extraction"], run_conditioning),
            StageSpec("flow_direction", "D8 flow direction", [paths["flow_direction"]], fp_dir,
                      ["conditioning"], run_direction),
            StageSpec("flow_accumulation", "D8 flow accumulation", [paths["flow_accumulation"]], fp_acc,
                      ["flow_direction"], run_accumulation),
            StageSpec("drainage", "drainage network", [paths["drainage_network"]], fp_drain,
                      ["flow_accumulation", "flow_direction"], run_drainage),
            StageSpec("hand", "HAND", [paths["hand"]], fp_hand, ["conditioning", "drainage"], run_hand),
        ):
            executor.add(spec)

        log.info("Stages 2-5: extraction, conditioning, flow routing, drainage, HAND")
        executor.ensure("hand")
        executor.collect_reused(list(executor.specs))
        cached_ext = executor.result("dem_extraction") or {}
        if "dem_extraction" in executor.reused and cached_ext.get("pilot_signature") != pilot_sig:
            warnings.add(
                "The pilot area changed since the reused DEM-extraction stage ran; its pilot-cell statistics refer "
                "to the previous pilot area (the final map always uses the current one)."
            )

        # --- Stages 6-7 ------------------------------------------------------
        log.info("Stages 6-7: HAND threshold (%g %s), clipping and export", p.hand_threshold, config.inputs.vertical_units)
        t_final = time.monotonic()
        final = build_final_mask(paths["hand"], domain.pilot, p.hand_threshold,
                                 paths["potential_floodplain"], config.inputs.vertical_units, cell)
        executor.timings["floodplain"] = {"status": "run", "seconds": round(time.monotonic() - t_final, 2)}
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
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        summary["error"] = "KeyboardInterrupt"
        log.error("Interrupted. Completed stages are kept; set execution.resume: true and run again to continue.")
        raise
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        log.error("Processing failed: %s", exc)
        if executor is not None and executor.manifest.entries:
            log.error(
                "Completed stages are kept in %s. Fix the problem and set execution.resume: true to continue "
                "from the last completed stage instead of starting over.", plan.paths["stage_manifest"],
            )
        raise
    finally:
        finished = datetime.now(timezone.utc)
        summary["finished_utc"] = finished.isoformat()
        summary["duration_seconds"] = (finished - started).total_seconds()
        try:
            summary["software_versions"] = collect_versions(locals().get("wbt_version"))
            if "runner" in locals():
                summary["whitebox_calls"] = runner.calls
            if executor is not None:
                res = {name: executor.result(name) for name in executor.specs}
                if res.get("dem_extraction"):
                    summary["stages"]["dem_extraction"] = res["dem_extraction"]
                if res.get("conditioning"):
                    summary["stages"]["conditioning"] = {"method": p.conditioning_method, **res["conditioning"]}
                if res.get("flow_direction") or res.get("flow_accumulation"):
                    summary["stages"]["flow_routing"] = {
                        "algorithm": p.flow_algorithm,
                        "direction": res.get("flow_direction"), "accumulation": res.get("flow_accumulation"),
                    }
                if res.get("drainage"):
                    summary["stages"]["drainage"] = res["drainage"]
                if res.get("hand"):
                    summary["stages"]["hand"] = res["hand"]
                summary["stages_reused_from_previous_run"] = list(executor.reused)
                summary["performance"] = {
                    "total_seconds": round(time.monotonic() - t_start, 2),
                    "stages": executor.timings,
                    "python_peak_rss_gib": resources.gib(resources.self_peak_rss_bytes()),
                    "whitebox_peak_rss_gib_sampled": resources.gib(runner.peak_rss_bytes) if "runner" in locals() else None,
                }
        except Exception:  # pragma: no cover - never mask the primary error
            pass
        summary["warnings"] = warnings.messages
        summary["limitations"] = LIMITATIONS
        write_summary(plan.paths["processing_summary"], summary)

    log.info("Done in %s. Final product: %s", resources.format_duration(time.monotonic() - t_start),
             plan.paths["potential_floodplain"])
    return summary
