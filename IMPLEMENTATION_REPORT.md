# Implementation report

## 0. Important: pre-existing files were overwritten

The project directory already contained an earlier implementation (files dated 2026-10-03). This session
wrote its files **without first listing the directory**, using shell redirection that overwrites silently.
The earlier version's `egg-info/SOURCES.txt` shows it had these modules, all of which were replaced by the
versions in this report: `__init__, cli, config, delineation, hand, hydrology, inputs, outputs, pipeline,
preprocessing, rasters, wbt` (confirmed by that file list), and `pyproject.toml` (also listed there).
**Probably** replaced as well, inferred rather than confirmed because only package modules are listed: `tests/conftest.py`
(the surviving old test uses a `make_dem` fixture that my `conftest.py` does not define), `tests/test_config.py`,
`test_inputs.py`, `test_delineation.py`, `test_pipeline.py`, `requirements.txt`, `config/example.yaml`, and
`scripts/make_synthetic_example.py` (the surviving old `config/synthetic_example.yaml` refers to a script of that name).
I cannot tell which of these existed. There was no git history or backup in the directory, so overwritten content
cannot be recovered from here.

Left untouched: `src/floodplain_delineation/errors.py`, `src/floodplain_delineation/__main__.py`,
`tests/test_hydrology_hand.py`, `config/synthetic_example.yaml`, `data/synthetic_dem.tif`,
`data/synthetic_pilot_area.gpkg`, `outputs/synthetic_demo/` (a complete earlier run), `src/*.egg-info`.

Consequences to be aware of:
* `tests/test_hydrology_hand.py` belongs to the earlier design (imports `floodplain_delineation.errors`,
  `extract_drainage_network`, `validate_conditioned_dem`, a `make_dem` fixture) and **cannot run** against this
  code; it fails at import and breaks collection of a plain `pytest` run. It is excluded below with `--ignore`.
  It was neither deleted nor edited. Decide whether to delete it or port its cases.
* `outputs/synthetic_demo/` was produced by the earlier code. Nothing here relies on it or reproduces it.
* If the earlier implementation is wanted back, it must come from the user's own backup/VCS.

## 1. What was implemented

All stages of the specification: input validation and spatial preparation, conditioning, D8 flow
direction/accumulation, drainage extraction, HAND, thresholding, clipping/export, summary/log/GPKG, CLI,
tests, documentation. Excluded features (validation, classifiers, TWI/REM, calibration, GUI, etc.) were not implemented.

## 2. WhiteboxTools operations (verified against the installed v2.0.0 binary's own help)

`BreachDepressions` (default) or `FillDepressionsWangAndLiu --fix_flats`, `D8Pointer`,
`D8FlowAccumulation --pntr --out_type=cells`, `ExtractStreams --zero_background`, `ElevationAboveStream`.
Behaviours established empirically and then covered by tests:
* `ExtractStreams` marks accumulation **strictly greater** than its threshold -> `N - 0.5` is passed to get `>= N`.
* D8 pointer codes: east=2, south=8, south-east=4 (default non-ESRI scheme).
* `ElevationAboveStream` equals an independent numpy trace of the D8 pointer to the first drainage cell (max difference 0.0 on the synthetic test DEM).
* The binary returns 0 for unknown flags, so each flag is checked against the tool's help text before running.

## 3. Supported configuration parameters

See README section 5 (`inputs.*`, `processing.*`, `outputs.*`, `execution.*`). Required scientific parameters have no defaults; unknown keys are rejected.

## 4. Tests executed and actual results

Command: `pytest tests --ignore=tests/test_hydrology_hand.py`
Result: **85 passed, 1 warning** (the warning comes from writing a GPKG without a CRS on purpose in a test).
`pytest -m wbt`: **7 passed** with the real WhiteboxTools executable, 0 skipped. Machine: 1 CPU, Python 3.12.3,
WhiteboxTools v2.0.0 binary. Covered: valid/invalid YAML, missing/unknown/invalid parameters, missing inputs, DEM
without CRS/nodata/rotated/multiband, empty/invalid/non-polygon/CRS-less pilot geometry, CRS transformation, empty
intersection, grid-aligned windows, buffer in projected and geographic CRS, nodata preservation, threshold behaviour
(spec example `[[0,2,5],[7,9,3]]`, threshold 5 -> `[[1,1,1],[0,0,1]]`), nodata kept in the mask, output metadata,
threshold conversion, failure propagation (fake executable: exit status, error text, missing output, undocumented flag),
overwrite conflicts, output/input collision, CLI exit codes, and end-to-end runs including the HAND cross-check.

An end-to-end run of `config/example.yaml` on the synthetic data exited 0. It emitted warnings that the 5 km buffer is truncated
by the (small) synthetic DEM on all sides and that 11.8% of valid cells have no associated drainage cell. These numbers describe
synthetic data only.

## 5. Known limitations / unresolved decisions

* Only D8 routing; `fill` is less reliable than `breach` in the tested WhiteboxTools build (see README section 9).
* `FillDepressions` (the canonical tool) was not used because it crashed on 3 of 5 synthetic DEMs; root cause unknown.
* Geographic CRS + `area_km2` threshold: cell area is geodesic at the central latitude (approximation, warned, variation quantified).
* HAND is nodata for cells whose flow path leaves the extent, and those cells stay nodata in the final mask. This is a design choice, documented, not a claim that they are non-floodplain.
* Drainage length is approximate (D8 step lengths).
* No validation, no real-data runs, no caching of intermediates.
* Interior nodata holes in the DEM are not filled by the workflow; with `fill` they trigger an explicit error if the tool corrupts them.

## 6. Version 0.2.0 - hardening for very large rasters

Trigger: a ~917 M-cell DEM failed in `D8Pointer` (`Error converting TAG(34737) to ASCII`, WhiteboxTools 2.4.0).

**Changes**
* Bugs fixed: `validate_input_files` had its whole body inside a docstring (no input validation ran; 2 tests were failing);
  `set_max_procs(18)` acted on a discarded object (no effect, hard-coded); `source_crs` stored a CRS object instead of a string;
  `pilot_area` was typed `Path` but stored as `str`/`None`; the summary printed `"None"` for a missing pilot area; `scripts/run_pipeline.py` import.
* A (robustness): WhiteboxTools raster outputs are rewritten by GDAL before reuse; per-stage resume with fingerprints
  (`stage_manifest.json`); version policy (`execution.whitebox_version`, warning for untested versions); `execution.max_procs`,
  `execution.scratch_directory`; partial/raw leftovers are removed at start; a child process is killed on interruption/timeout.
* B (memory): extraction, conditioning comparison, flow-direction/accumulation/drainage validation, drainage diagnostics, HAND
  statistics and the final mask are block-wise; GDAL cache fixed at 256 MB; relative tolerance for the float32 accumulation check.
* C (observability): per-stage timing and memory, heartbeat and streamed tool output, pre-flight estimate, breach-depth distribution.

**Measured here (synthetic data, 1 CPU, 3 GB RAM, WhiteboxTools 2.0.0 binary)**
* 131 tests pass (85 before + 46 new). Baseline of the uploaded zip: 2 failed (`test_missing_input_files`, `test_cli_reports_missing_input_with_nonzero_status`, both caused by the docstring bug) and 5 errors only because unzipping lost the executable bit of `tests/fake_wbt.py` (now set by `conftest.py`).
* Same synthetic 36 M-cell run, old vs new code: results identical (floodplain cells, HAND median); Python peak 1.11 GiB -> 0.57 GiB; wall time unchanged (~70 s, dominated by WhiteboxTools).
* Python-only stages on a 180 M-cell raster: peak 0.42 GiB, ~3 s per stage.

**Not verified**
* The failure itself was not reproduced (no 2.4.0 binary, too little RAM/disk in the development sandbox). That the GDAL rewrite avoids it follows from the observation that
  WhiteboxTools read a GDAL-written file of the same size; confirm on the target machine.
* No run at the 917 M-cell scale; WhiteboxTools memory and time at that size are unmeasured (the new log reports them).
* `--max_procs` is passed through; its effect per tool was not measured.
* Sampled WhiteboxTools memory can miss peaks between samples (every <= 5 s); tools shorter than that report "not sampled".

Not implemented (deferred by decision): alternative conditioning tools/limits, TauDEM/MPI, basin tiling.

## 7. Version 0.3.0 - REM per stream-link basin, global processing CRS, least-cost breach

Based on `CIDI_REM_IDW_por_bacia.md` plus two extra instructions: exclusive (non-nested) basins per river reach, and
`BreachDepressionsLeastCost` with configurable limits and a breach-then-fill option.

**State found at the start.** The working tree already contained an untested draft (`rem.py`, `reprojection.py`, changes in
`spatial.py`, `inputs.py`, `config.py`, `outputs.py`, `pipeline.py`, `preprocessing.py`, `rasters.py`) with no YAML, test or
documentation updates; the existing test suite could not even be collected, the REM stages were registered but never executed, and
the summary still used old stage names. The draft was reviewed and completed, not rewritten.

**Decisions (please confirm)**
* `Subbasins` instead of `Basins` (extra instruction overrides CIDI 5.1): `Basins` only delineates basins draining to the DEM edge.
  Behaviour of `Subbasins` was verified, not assumed: one unit per stream link, exclusive, drainage cells carry the link's ID, IDs
  positive but not contiguous, cells that reach no stream are NoData. It copies the NoData of the streams raster to its output
  (255 for the pipeline's uint8 drainage raster, which would erase basin 255), so it is given an int32 copy with NoData -9999.
* The CIDI contradicts itself on the reference DEM (sec. 2 step 2: initial DEM, "not the conditioned one"; REM-2: conditioned DEM
  by default). The explicit method definition was followed: the REM samples and subtracts the **unconditioned** DEM on the processing
  grid. HAND was not changed (sec. 12); the parenthetical "do not use the conditioned DEM for HAND" in sec. 2 contradicts sec. 12 and
  is left as an open question.
* "min and max configurable" was mapped to the real flags of `BreachDepressionsLeastCost`: `--dist` (maximum breach length, required,
  no default), `--max_cost`, `--min_dist`, `--flat_increment`. If other parameters were meant, only `config.py`/`preprocessing.py` change.
* "Breach then fill": two explicit steps (`BreachDepressionsLeastCost`, then `FillDepressionsWangAndLiu --fix_flats`), reported separately.
  The tool's own `--fill` was not used: in WhiteboxTools 2.0.0 it panics (`Error unwrapping 'output'`, exit 101) whenever a depression
  is left to fill (reproduced; unaffected by output name/dir, compression, verbosity). Not checked on 2.4.0.
* Sampling is per (block, basin), an extension of "one per block": with link-level basins two links meet inside one block, and a plain
  per-block rule would leave one of them without a sample. N = 1 keeps every valid drainage pixel.
* IDW: exact, all basin samples, float64, chunked; no KD-tree (an exact all-sample IDW cannot use neighbour queries) and no new dependency.
* Warp reproducibility: GDAL chooses the kernel width from the size of the destination window, so warped values depend slightly on the
  stripe height (1e-4 m with one window up to centimetres with a few hundred rows; the `gdalwarp -wm` behaviour). `tolerance=0` and an
  explicit source window were tried and did not remove it. The stripe height is therefore a fixed function of the grid width, independent
  of the memory budget, and part of the stage fingerprint.

**Changes.** `config.py` (processing CRS, resolution, resampling, REM, least-cost options), `spatial.py`/`inputs.py`/`reprojection.py`
(stage 0), `rem.py` (basins, sampling, IDW, REM), `preprocessing.py` (conditioning steps), `pipeline.py` (REM stages after the HAND map,
fingerprints, summary, limitations), `preflight.py`, YAMLs, README, `docs/methodology.md`, tests.

**Measured here (synthetic data, 1 CPU, 3 GB RAM, WhiteboxTools 2.0.0 binary)**
* 213 tests pass (131 before; 82 new).
* 16 M-cell geographic DEM -> Albers grid (12.7 M valid cells): whole run 62 s; reprojection 2.1 s; 388 basins, 21 688 samples;
  IDW 1.2e9 cell-sample evaluations in 21 s (5.6e7/s); Python peak 0.77 GiB.
* `BreachDepressionsLeastCost` also raises a few cells (152 cells, at most 0.063 m on a synthetic DEM).

**Not verified**
* Nothing was run at the 917 M-cell scale or on real data; IDW cost there depends on cells x samples per basin and was not extrapolated.
* WhiteboxTools 2.4.0 was not available: `Subbasins` semantics and the `--fill` panic were checked on the 2.0.0 binary only.
* No scientific validation of HAND or REM. The REM stage ordering after the HAND map means a REM failure never costs the HAND product.

