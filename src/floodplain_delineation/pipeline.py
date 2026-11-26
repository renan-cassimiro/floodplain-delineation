"""High-level orchestration of the workflow (Stages 0-7, optional REM, plus metadata).

Stage 1 (inputs, processing CRS and master grid) and the final HAND-threshold stage are cheap and always run.
Every other stage is resumable (see :mod:`stages` and :mod:`manifest`):

    dem_reprojection -> conditioning -> flow_direction -> flow_accumulation -> drainage -> hand
    flow_direction -> basins;  (basins, drainage, dem_reprojection) -> rem_samples
    (rem_samples, basins, dem_reprojection) -> rem_surface -> rem          [only with processing.rem.enabled]

The REM stages never feed the HAND stages: HAND and ``potential_floodplain.tif`` are identical with the REM on or off.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pyproj
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
from .preprocessing import condition_dem, conditioning_plan
from .rasters import GridInfo
from .reprojection import reproject_dem, warp_settings
from .rem import (
    BASINS_FLAGS, BASINS_TOOL, IDW_METHOD_ID, REM_METHOD_ID, SAMPLING_METHOD_ID, STREAMS_INPUT_NODATA, compute_rem,
    delineate_basins, interpolate_surface, load_samples, sample_drainage, save_samples,
)
from .spatial import cell_geometry
from .stages import StageExecutor, StageSpec
from .wbt import TESTED_WBT_VERSIONS, WhiteboxRunner, parse_version

log = get_logger("pipeline")

LIMITATIONS = [
    "The DEM is reprojected and resampled to the processing CRS/grid before any hydrological operation (unless it "
    "already is on it); elevation values change and results can differ from runs in the original CRS.",
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


REM_LIMITATIONS = [
    "The REM is a relative elevation product (DEM minus a surface interpolated from the elevations of sampled drainage "
    "cells); it is not an observed or hydraulic water surface and is not validated.",
    "HAND associates each cell with the drainage cell reached by its D8 path; the REM instead uses an euclidean IDW "
    "surface per stream-link basin. IDW does not guarantee longitudinal hydrological continuity and does not respect "
    "topographic barriers inside a basin.",
    "The REM depends on the initial reprojection/resampling and resolution of the DEM, on the basin units produced by "
    "WhiteboxTools Subbasins (hence on the conditioned DEM, the D8 pointer, the drainage threshold, the extent and the "
    "buffer), on the sampling interval, on the IDW exponent and on the DEM quality.",
    "The reference elevation is the unconditioned DEM on the processing grid (HAND uses the conditioned DEM). Negative REM "
    "values are preserved. Cells that drain to no stream inside the extent have no basin and are NoData in the REM.",
    "Euclidean distances are computed in the processing CRS; unless it is equidistant, they differ from ground distances.",
]


def _rem_summary(p: Any, stages: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    """Compact REM section of the processing summary (details are in ``stages``)."""
    if not p.rem.enabled:
        return {"enabled": False}
    st = lambda k, key: (stages.get(k) or {}).get(key)  # noqa: E731
    return {
        "enabled": True,
        "parameters": {
            "sampling_interval_pixels": p.rem.sampling_interval_pixels,
            "idw_power": p.rem.idw_power,
            "distance": "euclidean, in linear units of the processing CRS",
            "basins": f"WhiteboxTools {BASINS_TOOL}: one exclusive (non-nested) catchment per stream link",
            "reference_dem": "unconditioned DEM on the processing grid (dem_processing_crs.tif)",
        },
        "methods": {"sampling": SAMPLING_METHOD_ID, "interpolation": IDW_METHOD_ID, "rem": REM_METHOD_ID},
        "basins": st("basins", "basins"),
        "samples": st("rem_samples", "samples"),
        "basins_with_samples": st("rem_samples", "basins_with_samples"),
        "basins_without_samples": st("rem_surface", "basins_without_samples"),
        "rem_valid_cells": st("rem", "valid_cells"),
        "rem_negative_fraction": st("rem", "negative_fraction"),
        "products": {k: (str(paths[k]) if paths[k].exists() else None)
                     for k in ("basins", "rem_samples", "rem_drainage_surface", "rem")},
    }


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
        cond_steps = conditioning_plan(p)
        needed_tools = [t for t, _ in cond_steps] + ["D8Pointer", "D8FlowAccumulation", "ExtractStreams", "ElevationAboveStream"]
        if p.rem.enabled:
            needed_tools.append(BASINS_TOOL)   # only the REM needs Subbasins; HAND-only runs never call it
        runner.verify_tools(needed_tools)
        log.info("Using %s", wbt_version)
        _check_wbt_version(config.execution.whitebox_version, wbt_version, warnings)

        # --- Stage 1 ---------------------------------------------------------
        log.info("Stage 1: input validation and spatial preparation")
        dem = read_dem_info(config.inputs.dem, warnings)
        domain = prepare_spatial_domain(dem, config, warnings)
        mgrid = domain.grid
        cell = cell_geometry(mgrid.pcrs, mgrid.transform, domain.extent_bounds)
        write_processing_area(plan.paths["processing_area"], domain, mgrid.crs)
        win_w, win_h = mgrid.width, mgrid.height
        grid = GridInfo(win_h, win_w, mgrid.transform, mgrid.crs)
        proc_crs = domain.processing_crs
        log.info("Processing CRS: %s (%s, %s); master grid %d x %d cells at %.6g x %.6g %s.",
                 proc_crs.name, proc_crs.authority, proc_crs.unit_name, win_w, win_h, mgrid.res_x, mgrid.res_y, proc_crs.unit_name)
        summary["spatial"] = {
            "source_dem_crs": dem.crs.to_string(),
            "source_dem_resolution_crs_units": list(dem.resolution),
            "processing_crs": proc_crs.describe(),
            "master_grid": mgrid.describe(),
            "resolution": domain.resolution_info,
            "dem_resampling": config.processing.dem_resampling,
            "pilot_source_crs": domain.source_crs,
            "output_crs": mgrid.crs.to_string(),
            "reprojection_note": (
                "The DEM and the pilot geometry are brought to the processing CRS/grid before any hydrological "
                "operation. The source DEM file is never modified. Reprojection/resampling changes elevation values "
                "and can change hydrological results compared with runs in the original CRS."
            ),
            "cell_size_m": [cell.dx_m, cell.dy_m],
            "cell_area_km2": cell.area_km2,
            "vertical_units": config.inputs.vertical_units,
            "dem_nodata": dem.nodata,
            "dem_dtype": dem.dtype,
            "pilot_features": domain.n_features,
            "pilot_coverage_by_dem": domain.coverage_fraction,
            "processing_extent_bounds": list(domain.extent_bounds),
            "requested_extent_bounds": list(domain.requested_bounds),
            "source_window": None if domain.window is None else [
                int(domain.window.col_off), int(domain.window.row_off), int(domain.window.width), int(domain.window.height)],
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
            "dem_reprojection",
            {
                **common,
                "dem": file_signature(config.inputs.dem),
                "processing_crs": mgrid.crs.to_wkt(),
                "grid": {"transform": list(tuple(mgrid.transform))[:6], "width": win_w, "height": win_h},
                "resolution": [mgrid.res_x, mgrid.res_y],
                "resampling": "none" if mgrid.identical_to_source else config.processing.dem_resampling,
                "warp": None if mgrid.identical_to_source else warp_settings(),
                "libs": {"gdal": rasterio.__gdal_version__, "rasterio": rasterio.__version__,
                         "proj": pyproj.proj_version_str, "numpy": np.__version__},
            },
        )
        fp_cond = fingerprint(
            "conditioning",
            {**common, "method": p.conditioning_method, "steps": cond_steps, "wbt": wbt_id},
            [fp_ext],
        )
        fp_dir = fingerprint("flow_direction", {**common, "wbt": wbt_id}, [fp_cond])
        fp_acc = fingerprint("flow_accumulation", {**common, "wbt": wbt_id}, [fp_dir])
        fp_drain = fingerprint("drainage", {**common, "wbt": wbt_id, "threshold": thr}, [fp_acc, fp_dir])
        fp_hand = fingerprint("hand", {**common, "wbt": wbt_id}, [fp_cond, fp_drain])

        manifest = StageManifest(paths["stage_manifest"])
        if config.execution.resume:
            manifest.load()
        executor = StageExecutor(manifest, warnings, config.execution.resume, tool_peak=lambda: runner.peak_rss_bytes)

        def run_reproject() -> Dict[str, Any]:
            info = reproject_dem(dem, domain, paths["dem_processing_crs"], config.processing.dem_resampling, warnings)
            info["pilot_signature"] = pilot_sig
            return info

        def run_conditioning() -> Dict[str, Any]:
            return condition_dem(runner, p.conditioning_method, cond_steps, paths["dem_processing_crs"],
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
            StageSpec("dem_reprojection", "DEM reprojection to the processing grid", [paths["dem_processing_crs"]],
                      fp_ext, [], run_reproject),
            StageSpec("conditioning", f"hydrological conditioning ({p.conditioning_method})",
                      [paths["conditioned_dem"]], fp_cond, ["dem_reprojection"], run_conditioning),
            StageSpec("flow_direction", "D8 flow direction", [paths["flow_direction"]], fp_dir,
                      ["conditioning"], run_direction),
            StageSpec("flow_accumulation", "D8 flow accumulation", [paths["flow_accumulation"]], fp_acc,
                      ["flow_direction"], run_accumulation),
            StageSpec("drainage", "drainage network", [paths["drainage_network"]], fp_drain,
                      ["flow_accumulation", "flow_direction"], run_drainage),
            StageSpec("hand", "HAND", [paths["hand"]], fp_hand, ["conditioning", "drainage"], run_hand),
        ):
            executor.add(spec)

        rem_cfg = p.rem
        if rem_cfg.enabled:
            libs = {"numpy": np.__version__}
            fp_basins = fingerprint(
                "basins",
                {**common, "wbt": wbt_id, "tool": BASINS_TOOL, "flags": BASINS_FLAGS,
                 "inputs": ["flow_direction", "drainage_network"], "streams_nodata": STREAMS_INPUT_NODATA},
                [fp_dir, fp_drain],
            )
            fp_samples = fingerprint(
                "rem_samples",
                {**common, "method": SAMPLING_METHOD_ID, "interval": rem_cfg.sampling_interval_pixels, "libs": libs,
                 "crs": mgrid.crs.to_wkt(), "resolution": [mgrid.res_x, mgrid.res_y]},
                [fp_basins, fp_drain, fp_ext],
            )
            fp_surface = fingerprint(
                "rem_surface", {**common, "method": IDW_METHOD_ID, "power": rem_cfg.idw_power, "libs": libs},
                [fp_samples, fp_basins, fp_ext],
            )
            fp_rem = fingerprint("rem", {**common, "method": REM_METHOD_ID, "libs": libs}, [fp_surface, fp_ext])

            def run_basins() -> Dict[str, Any]:
                return delineate_basins(runner, paths["flow_direction"], paths["drainage_network"], paths["basins"],
                                        work, grid, warnings)

            def run_samples() -> Dict[str, Any]:
                table, stats = sample_drainage(paths["drainage_network"], paths["dem_processing_crs"], paths["basins"],
                                               rem_cfg.sampling_interval_pixels)
                save_samples(paths["rem_samples"], table, stats)
                return stats

            def run_surface() -> Dict[str, Any]:
                table = load_samples(paths["rem_samples"])
                return interpolate_surface(table, paths["basins"], paths["dem_processing_crs"],
                                           paths["rem_drainage_surface"], grid, rem_cfg.idw_power, warnings)

            def run_rem() -> Dict[str, Any]:
                return compute_rem(paths["dem_processing_crs"], paths["rem_drainage_surface"], paths["rem"], grid, warnings)

            for spec in (
                StageSpec("basins", f"stream-link basin delineation (WhiteboxTools {BASINS_TOOL})", [paths["basins"]],
                          fp_basins, ["flow_direction", "drainage"], run_basins),
                StageSpec("rem_samples", "REM drainage sampling", [paths["rem_samples"]], fp_samples,
                          ["basins", "drainage", "dem_reprojection"], run_samples),
                StageSpec("rem_surface", "REM IDW surface (per basin)", [paths["rem_drainage_surface"]], fp_surface,
                          ["rem_samples", "basins", "dem_reprojection"], run_surface),
                StageSpec("rem", "REM (DEM - IDW surface)", [paths["rem"]], fp_rem, ["rem_surface", "dem_reprojection"], run_rem),
            ):
                executor.add(spec)

        log.info("Stages 0, 2-5: reprojection, conditioning, flow routing, drainage, HAND")
        executor.ensure("hand")

        # --- Stages 6-7 ------------------------------------------------------
        log.info("Stages 6-7: HAND threshold (%g %s), clipping and export", p.hand_threshold, config.inputs.vertical_units)
        t_final = time.monotonic()
        final = build_final_mask(paths["hand"], domain.pilot, p.hand_threshold,
                                 paths["potential_floodplain"], config.inputs.vertical_units, cell)
        executor.timings["floodplain"] = {"status": "run", "seconds": round(time.monotonic() - t_final, 2)}
        summary["stages"]["floodplain"] = {
            "hand_threshold": p.hand_threshold, "hand_threshold_units": config.inputs.vertical_units, **final,
        }

        # --- REM (independent product; never feeds HAND; runs after the HAND map is complete) -------------
        if rem_cfg.enabled:
            log.info("REM: stream-link basins, drainage sampling (every %d px), IDW (power %g) per basin, DEM minus surface",
                     rem_cfg.sampling_interval_pixels, rem_cfg.idw_power)
            executor.ensure("rem")

        executor.collect_reused(list(executor.specs))
        cached_ext = executor.result("dem_reprojection") or {}
        if "dem_reprojection" in executor.reused and cached_ext.get("pilot_signature") != pilot_sig:
            warnings.add(
                "The pilot area changed since the reused DEM-reprojection stage ran; its pilot-cell statistics refer "
                "to the previous pilot area (the final map always uses the current one)."
            )

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
                if res.get("dem_reprojection"):
                    summary["stages"]["dem_reprojection"] = res["dem_reprojection"]
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
                for name in ("basins", "rem_samples", "rem_surface", "rem"):
                    if res.get(name):
                        summary["stages"][name] = res[name]
                summary["stages_reused_from_previous_run"] = list(executor.reused)
                summary["performance"] = {
                    "total_seconds": round(time.monotonic() - t_start, 2),
                    "stages": executor.timings,
                    "python_peak_rss_gib": resources.gib(resources.self_peak_rss_bytes()),
                    "whitebox_peak_rss_gib_sampled": resources.gib(runner.peak_rss_bytes) if "runner" in locals() else None,
                }
        except Exception:  # pragma: no cover - never mask the primary error
            pass
        summary["rem"] = _rem_summary(p, summary.get("stages", {}), plan.paths)
        summary["warnings"] = warnings.messages
        summary["limitations"] = LIMITATIONS + (REM_LIMITATIONS if p.rem.enabled else [])
        write_summary(plan.paths["processing_summary"], summary)

    log.info("Done in %s. Final product: %s", resources.format_duration(time.monotonic() - t_start),
             plan.paths["potential_floodplain"])
    return summary
