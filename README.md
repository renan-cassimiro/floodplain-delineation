# floodplain_delineation

Python/YAML workflow that derives a **preliminary potentially inundatable floodplain map** from a
DEM using Height Above Nearest Drainage (HAND), with WhiteboxTools as the hydrological engine.

> **Status of the results.** The test suite and the example below use *synthetic* data only.
> No claim is made about performance on real floodplains: the workflow has not been run on, or
> validated against, any real dataset.

## 1. Overview

**Objective.** Map terrain that is potentially subject to riverine inundation, approximated by
relative elevation to a drainage network derived from the DEM:

    HAND(x) = Z(x) - Z_drain(x)
    potential floodplain = valid cells where HAND <= hand_threshold

**Intended use.** Exploratory geomorphological/hydrological research (e.g. Amazonian floodplains);
no basin-specific parameter is hard-coded.

**Second product: REM.** Optionally (`processing.rem.enabled`) the pipeline also writes a *Relative Elevation Model*:
the DEM minus a surface interpolated (IDW) from the elevations of sampled drainage pixels, **separately for the exclusive
catchment of every stream link** (section 6.1). It is independent of HAND and never feeds it.

**What the product is not.** It is a topographic approximation. It is not observed inundation, a
hydraulic simulation, a flood-recurrence map, or a validated floodplain boundary. HAND thresholds
are not water levels.

## 2. Requirements and installation

* Python >= 3.9 (developed and tested on 3.12).
* Python packages: `numpy rasterio geopandas shapely>=2 pyproj pyyaml whitebox` (+ `pytest`).
* WhiteboxTools executable. `pip install whitebox` ships/downloads it. It is located, in order, from
  `execution.whitebox_executable`, the `WBT_EXE` environment variable, the `whitebox` package, then `PATH`.

```bash
pip install -r requirements.txt
pip install -e .            # optional: installs the `floodplain-delineation` command
```

## 3. Input specification

| Input | Requirements |
|---|---|
| **DEM** | Single band; valid CRS (projected or geographic); **nodata value defined**; north-up, non-rotated grid. Vertical units are declared in the config (`inputs.vertical_units`) and are the unit of the HAND threshold. |
| **Pilot area** | GeoPackage with >= 1 polygon feature (`inputs.pilot_layer` optional). Must have a CRS. Invalid polygons are repaired (`make_valid`) with a warning; several polygons are dissolved into one. Non-polygon/empty features are dropped with a warning. |

* **The DEM is reprojected once**, in stage 0, to the processing CRS (`processing.crs`, default `ESRI:102033`, South America
  Albers Equal Area Conic) at `processing.target_resolution` (default: derived by GDAL from the source DEM and recorded). The
  result, `dem_processing_crs.tif`, is the master grid of the whole run: conditioning, D8, drainage, HAND, basins, sampling,
  IDW and REM all use it. If the DEM already is in that CRS and no resolution is requested, it is only cropped (exact copy, no
  resampling). The source file is never modified. Elevation values change when resampling happens; this is warned and recorded.
* The processing CRS must be **projected** (linear units); geographic CRSs are rejected because the IDW uses euclidean distances.
  `ESRI:102033` keeps its ESRI authority: it is never rewritten as an EPSG code, and an unknown identifier is an error.
* The pilot geometry is reprojected to the processing CRS. The pilot area must intersect the DEM footprint (the footprint is
  transformed with densified edges), otherwise the run fails with a clear error.
* Nodata is never interpolated across: after bilinear/cubic resampling, a cell is valid only if the *nearest* source cell is valid.
* Integer DEMs are stored as float32 in `dem_processing_crs.tif`. A NaN nodata is replaced by the explicit sentinel `-9999`.
* Area-based drainage thresholds use the cell area of the processing CRS (exact for an equal-area CRS).

## 4. Running

```bash
python scripts/make_synthetic_example.py --out-dir data     # synthetic DEM + pilot area (optional)
python -m floodplain_delineation.cli --config config/example.yaml
python -m floodplain_delineation.cli --help
```

Exit status: `0` success, `1` expected failure (config/input/processing/output), `2` usage error, `3` unexpected error.
Relative paths in the YAML are resolved against the **current working directory**.

## 5. Configuration guide

See `config/example.yaml`. Every scientific parameter is **required**; unknown keys are rejected.

| Key | Required | Units / values | Notes |
|---|---|---|---|
| `inputs.dem`, `inputs.pilot_area` | yes | paths | `pilot_area: null` processes the whole DEM footprint |
| `inputs.pilot_layer` | no | layer name or `null` | warning if the GPKG has several layers and none is named |
| `inputs.vertical_units` | yes | `meters`, `centimeters`, `decimeters`, `feet`, `us_survey_feet` | documents the unit of DEM and HAND threshold |
| `processing.crs` | no (`ESRI:102033`) | projected CRS identifier | CRS of the whole processing; section 3 |
| `processing.target_resolution` | no (`null`) | `> 0`, CRS units | `null`: derived from the source DEM by GDAL; the value and its origin are recorded |
| `processing.dem_resampling` | no (`bilinear`) | `near`, `bilinear`, `cubic`, `cubic_spline`, `lanczos` | continuous data only |
| `processing.buffer_distance` | yes | `>= 0`, in `buffer_units` | context around the pilot area, measured in the processing CRS |
| `processing.buffer_units` | no (default `meters`) | `meters`, `cells` | |
| `processing.conditioning.method` | yes | `breach`, `fill`, `breach_least_cost` | see section 6 |
| `processing.conditioning.breach_least_cost.max_distance_cells` | with `breach_least_cost` | whole number `>= 1` | `--dist`: longest breach path, in cells; **no default** (it bounds how much terrain can be modified) |
| `processing.conditioning.breach_least_cost.max_cost` | no | `> 0` or `null` | `--max_cost`; `null`: tool default |
| `processing.conditioning.breach_least_cost.minimize_distance` | no (`false`) | bool | `--min_dist` |
| `processing.conditioning.breach_least_cost.flat_increment` | no | `> 0` or `null` | `--flat_increment`; `null`: tool default |
| `processing.conditioning.fill_remaining` | no (`false`) | bool | `breach_least_cost` only: after the breach, fill what is left (second step) |
| `processing.rem.enabled` | no (`false`) | bool | compute the REM (section 6.1); `Subbasins` runs only then |
| `processing.rem.sampling_interval_pixels` | with `rem.enabled` | whole number `>= 1` | block size N of the drainage sampling; `1` keeps every drainage pixel |
| `processing.rem.idw_power` | no (`2.0`) | `> 0`, finite | IDW exponent |
| `processing.flow.algorithm` | yes | `d8` | the only supported algorithm |
| `processing.drainage.threshold_type` | yes | `cells`, `area_km2` | |
| `processing.drainage.threshold_value` | yes | `> 0` (whole number for `cells`) | minimum contributing cells / km2 |
| `processing.hand.threshold` | yes | `>= 0`, DEM vertical units | |
| `outputs.directory` | yes | path | |
| `outputs.intermediates`, `outputs.final` | no (`intermediate`, `final`) | relative names | must differ, no `..` |
| `outputs.save_*` | no (`true`) | bool | `false` deletes that intermediate after the run (`save_basins`, `save_rem_samples`, `save_rem_drainage_surface` for the REM) |
| `outputs.rem` | no (`rem`) | relative name | sub-directory of `rem.tif` and the IDW surface |
| `execution.overwrite` | no (`false`) | bool | start from scratch; conflicts are checked **before** any processing |
| `execution.resume` | no (`false`) | bool | reuse up-to-date stages of a previous run (section 11). Mutually exclusive with `overwrite` |
| `execution.verbose` | no (`true`) | bool | DEBUG vs INFO logging |
| `execution.whitebox_executable` | no | path | |
| `execution.whitebox_version` | no | e.g. `"2.4.0"` | exact WhiteboxTools version required; without it a warning is recorded for versions not covered by the tests |
| `execution.max_procs` | no | integer >= 1 | passed to WhiteboxTools as `--max_procs`; many of its tools are single-threaded, so the effect depends on the tool |
| `execution.heartbeat_seconds` | no (`300`) | seconds >= 1 | interval of the progress message while a long WhiteboxTools run is active |
| `execution.scratch_directory` | no | path | fast local disk for the *raw* WhiteboxTools outputs (see section 11) |

## 6. Processing methodology

0. **Processing grid and DEM reprojection (stage 0).** The processing extent (pilot + buffer, intersected with the densified DEM
   footprint) is turned into a grid in `processing.crs`: resolution from the configuration or derived (GDAL
   `calculate_default_transform`, square cells), origin snapped to a multiple of the resolution. The DEM is warped onto it with
   `processing.dem_resampling` in fixed-height stripes (memory does not depend on the raster size; the stripe height is part of
   the stage fingerprint). Before/after statistics, the resampling and the nodata rule are recorded. GDAL derives the kernel width
   from the size of the window it warps, so values can differ by a few centimetres between different stripe heights (about 1e-4 m
   when one window covers the grid); fixing the stripe height keeps runs reproducible whatever the memory budget.
1. **Spatial preparation.** Pilot geometry is validated/repaired, reprojected to the processing CRS, dissolved, intersected with the
   footprint, buffered. Sides where the DEM edge truncates the requested buffer are reported. *A buffer cannot guarantee
   correct upstream hydrology if the DEM does not extend far enough beyond the area of interest.*
2. **Conditioning** (`dem_processing_crs.tif` is the unconditioned copy; the source DEM is never modified).
   * `breach` -> `BreachDepressions` (Lindsay 2016). Lowers terrain along least-cost paths.
   * `breach_least_cost` -> `BreachDepressionsLeastCost` with the configured `--dist`, `--max_cost`, `--min_dist`, `--flat_increment`.
     With `fill_remaining: true` a second step, `FillDepressionsWangAndLiu --fix_flats`, fills the depressions the bounded breach could
     not cross. The tool's own `--fill` flag is **not** used (section 9). Both steps are reported separately (cells modified, max
     decrease/increase). Note that the least-cost breach also *raises* a few cells slightly (flat-area increment); measured on a synthetic
     DEM: 152 cells, at most 0.063 m.
   * `fill` -> `FillDepressionsWangAndLiu --fix_flats`. Raises terrain inside depressions, which can create artificial flats.
   The result is validated: same grid, finite values, and an **identical nodata footprint**; any difference is an error. The
   summary also gives the distribution of elevation decreases (approximate p50/p95/p99 and the number of cells lowered by more than
   1, 5, 10, 25 and 50 vertical units), because an unbounded breach can cut tens of metres.
3. **Flow routing (D8 only).** `D8Pointer` -> `D8FlowAccumulation` (from the pointer, output in cells; each cell counts itself).
4. **Drainage network.** `ExtractStreams`. Cells with accumulation **>= N** are drainage. WhiteboxTools marks cells strictly
   *greater* than its threshold, so `N - 0.5` is passed (accumulation is integer valued; verified by tests).
   For `area_km2`: `N = ceil(area / cell_area)` using the actual cell area. Reported: threshold requested and effective
   (cells and km2), drainage cells, % of the processing area, approximate length (sum of D8 step lengths). These are
   descriptive and never used to tune the threshold.
5. **HAND.** `ElevationAboveStream`: elevation above the nearest **downslope** drainage cell along the D8 path. Drainage
   cells = 0. Cells whose path leaves the extent never reach drainage and are **nodata** (counted and warned).
   Min/max/mean/std/percentiles of valid cells and the count of negative values are reported.
6. **Mask.** `1` where `HAND <= threshold`, `0` where `HAND > threshold`, `255` nodata (never reclassified).
7. **Clipping.** Cropped to the pilot window on the same grid; cells whose centre lies outside the polygon become nodata.

### 6.1 Relative Elevation Model (REM)

HAND follows each cell's D8 path to the nearest downslope drainage cell. The REM instead subtracts a *surface* built from
drainage elevations, per basin. Both are computed; neither depends on the other.

1. **Basins (`basins.tif`).** WhiteboxTools `Subbasins` on the D8 pointer and the drainage network: **one exclusive catchment
   per stream link** (segment between confluences/heads), *not nested* (a cell belongs to exactly one unit, the drainage cells of
   a link carry its ID). Behaviour checked on synthetic data with the installed binary. IDs are positive but **not contiguous**;
   cells whose D8 path leaves the extent without reaching the network have no ID (NoData, counted). `Basins` was not used: it only
   delineates the basins that drain to the DEM edge. The pipeline hands `Subbasins` an int32 copy of the drainage raster with NoData
   `-9999`, because the tool gives its output the NoData of the streams raster (the pipeline's own uint8 raster has 255, which would
   erase basin 255). Validated: IDs positive integers, every drainage cell has an ID, no ID on DEM-nodata cells, no basin without a
   drainage cell (warned). A single unit is used as is. The units are as coarse as the drainage threshold makes the network.
2. **Sampling (`rem_samples.npz`).** The grid is divided into N x N pixel blocks anchored at the grid origin
   (`sampling_interval_pixels`); in each block, **for each basin that has valid drainage pixels there**, the pixel closest to the
   block centre is kept (ties: row, then column; independent of reading order). With N = 1 every valid drainage pixel is kept.
   The per-basin rule guarantees that every basin crossing the network gets at least one sample, which a plain "one per block"
   rule cannot (two links meet inside one block). Elevation = the **unconditioned** DEM on the processing grid; nodata pixels are
   never sampled.
3. **IDW per basin (`rem/rem_drainage_surface.tif`).** `Z(x) = sum(w_i z_i) / sum(w_i)`, `w_i = d(x, x_i)^-p`, euclidean distance
   in processing-CRS units, using **all samples of the cell's own basin** and nothing from other basins; no radius, neighbour limit
   or truncation. A cell on a sample takes its elevation exactly. Evaluated exactly in float64 in chunks (no global distance
   matrix; no KD-tree, because an exact all-sample IDW cannot use neighbour queries). Basins without samples give NoData
   (counted, never invented).
4. **REM (`rem/rem.tif`).** `REM = DEM - surface`, float32, **negative values preserved**; NoData where the DEM, the basin or the
   surface has none.

*Cost.* IDW cost is the sum over basins of (cells x samples) and is reported (`pair_evaluations`, per second). Measured here on
one CPU thread: 1.2e9 evaluations in 21 s (about 5.6e7 per second; 16 M-cell DEM, 388 basins, 21 688 samples). It is not extrapolated
to larger rasters; large basins with long links and a small interval make it grow quickly, so raise the interval before anything
else.

*Reference DEM.* The REM uses the unconditioned DEM; HAND uses the conditioned one (unchanged). The method description asks for the
initial DEM, while one line of the task statement says the opposite; the explicit method definition was followed.

## 7. Outputs

```
outputs/
├── intermediate/
│   ├── dem_processing_crs.tif      unconditioned DEM on the processing grid (master grid; float32 for integer DEMs)
│   ├── conditioned_dem.tif         conditioned DEM (processing extent)
│   ├── flow_direction.tif          D8 codes: 1=NE 2=E 4=SE 8=S 16=SW 32=W 64=NW 128=N, 0=no downstream neighbour
│   ├── flow_accumulation.tif       contributing cells
│   ├── drainage_network.tif        uint8: 1 drainage, 0 other, 255 nodata
│   ├── hand.tif                    HAND in DEM vertical units (nodata where no drainage association)
│   ├── basins.tif                  REM: stream-link basin IDs (int32; NoData -9999)
│   └── rem_samples.npz             REM: sampled drainage pixels (basin, row, col, x, y, z)
├── rem/
│   ├── rem_drainage_surface.tif    REM: IDW surface of drainage elevations (float32)
│   └── rem.tif                     REM: DEM - surface (float32, negatives kept)
├── final/
│   └── potential_floodplain.tif    uint8: 1 potential floodplain, 0 other, 255 nodata; clipped to the pilot area
├── stage_manifest.json             stage fingerprints for resume
├── processing_area.gpkg            pilot (clipped / full), requested and effective processing extents
├── processing_summary.json         parameters, CRS, resolution, thresholds, stats, versions, warnings, limitations
└── processing.log
```

All rasters share the processing grid (CRS, transform, shape) of `dem_processing_crs.tif`; the final mask is that grid cropped to the pilot window. Intermediates cover the
whole processing extent (pilot + buffer); only the final product is clipped. `processing_summary.json` is also written
(with `"status": "failed"` and the error) when a run fails after the overwrite check.

## 8. Scientific limitations

* DEM errors and limited vertical accuracy (vegetation/building bias, voids, striping) propagate directly into HAND.
* The drainage threshold strongly controls the network and therefore HAND; results are threshold-sensitive.
* HAND depends on the *derived* drainage network, not on observed rivers.
* Boundary effects and insufficient upstream context remain possible even with a buffer.
* HAND is not observed inundation; a HAND threshold is not a water level or return period.
* The map is a preliminary topographic approximation, **not a validated floodplain boundary**; no validation was performed.
* D8 routing over very flat terrain depends on WhiteboxTools' flat handling.
* The DEM is resampled to the processing grid (unless already on it); results differ from runs in the original CRS.
* REM: not an observed or hydraulic water surface; IDW ignores topographic barriers and longitudinal continuity inside a basin; it depends on the resolution, the basin units (hence the drainage threshold, extent and buffer), the sampling interval and the IDW exponent; euclidean distances are distorted unless the CRS is equidistant. Negative REM values are kept.

## 9. Known issues observed with the installed WhiteboxTools (v2.0.0 binary, tested here)

* `FillDepressions` terminated with exit status 101 (`Error unwrapping 'output'`) on 3 of 5 synthetic DEMs, and worked on 2;
  the cause was not identified. It is therefore **not used**; `fill` uses `FillDepressionsWangAndLiu`.
* On a DEM with an *interior* nodata hole, `FillDepressionsWangAndLiu` left an integer sentinel (-2147483648) as valid data in
  some hole cells and `FillDepressionsPlanchonAndDarboux` filled the hole with invented elevations. The pipeline's nodata-footprint
  check turns this into an explicit error. `breach` did not show the problem. These observations come from small synthetic tests
  and are not a general statement about WhiteboxTools.
* `BreachDepressionsLeastCost --fill` terminates with exit status 101 (`Error unwrapping 'output'`) whenever a depression is left to fill (reproduced on a synthetic DEM, independent of output name, directory, compression and verbosity; it works when nothing is left to fill). It is **not used**: `fill_remaining` runs `FillDepressionsWangAndLiu --fix_flats` as a second step. Same panic message as the `FillDepressions` issue above. Not checked on WhiteboxTools 2.4.0.
* `BreachDepressionsLeastCost` also raises a few cells (flat-area increment): 152 cells, at most 0.063 m on a synthetic DEM.
* WhiteboxTools `Subbasins` gives its output the NoData of the streams raster and numbers the units non-contiguously (see 6.1).
* WhiteboxTools ignores unknown flags silently and `whitebox.run_tool` returns 0 even if the tool panics. This project calls the
  binary directly, checks the exit status, checks every flag against the installed tool's help text, and verifies outputs exist.

## 10. Reproducibility

The summary stores the effective configuration, the SHA-256 of the YAML file, input paths, software versions (Python, numpy,
rasterio, GDAL, geopandas, shapely, pyproj, PyYAML, `whitebox` package, WhiteboxTools binary), every WhiteboxTools call (tool,
flags, exit status), UTC timestamps, warnings and limitations. Output organisation is fixed as shown above.
Stage fingerprints and the durations/peak memory of every stage and WhiteboxTools call are stored too (section 11).

## 11. Large rasters, resume and monitoring (v0.2.0)

Developed after a run on a ~917 M-cell DEM (18 993 x 48 316) that failed in `D8Pointer`.

**WhiteboxTools outputs are rewritten by GDAL.** Each raster written by WhiteboxTools (to a raw file in the working
directory) is copied block by block by GDAL into its final name (same grid, dtype, nodata, CRS, uncompressed GeoTIFF/BigTIFF)
before any other tool reads it. Reason: WhiteboxTools 2.4.0 could not read its own 3.7 GB `conditioned_dem.tif`
(`Error converting TAG(34737) to ASCII` in `whitebox-raster/.../ifd.rs`) although GDAL read it correctly, while it did read a
GDAL-written file of the same size. The root cause inside WhiteboxTools was **not** confirmed; the rewrite is a workaround
that follows from that observation and has been tested here only with the 2.0.0 binary (where the failure does not occur).

**Memory.** Validation, statistics and the final mask are computed on full-width row stripes (`rasters.BLOCK_BYTES`, 192 MB),
so Python memory does not grow with the raster. Measured here (synthetic data): 0.42 GiB peak at 180 M cells; the previous
whole-array code needed 1.11 GiB at 36 M cells (about 31 bytes per cell). WhiteboxTools itself still loads whole rasters;
its memory is sampled and reported per call. HAND percentiles come from a 100 000-bin histogram (error <= one bin width,
reported in the summary); other statistics are exact.

**Resume.** `execution.resume: true` reuses a stage when its fingerprint (relevant parameters, upstream fingerprints, DEM path/size/mtime,
window, WhiteboxTools and package version) matches `stage_manifest.json` and its output files exist with the recorded size.
Examples: changing `hand.threshold` re-runs only the final stage; changing the drainage threshold re-runs drainage and HAND;
changing the DEM, window, conditioning method or WhiteboxTools version re-runs what depends on it. A failed or interrupted run
keeps its completed stages. Limits: the DEM is identified by path/size/mtime, not content hash; hand-edited intermediates are
only detected through their size. With `save_*: false` a deleted intermediate is recomputed only if a later stage needs it.

**Monitoring.** The log shows, per stage, elapsed time and peak memory; during WhiteboxTools runs it prints a heartbeat every
`heartbeat_seconds` (elapsed time, tool memory) and streams the tool output to DEBUG. Before heavy work a pre-flight estimate
(raster size, disk needed vs free, available memory/container limit, CPUs) is logged and warnings are recorded. The conditioning
diagnostics include the distribution of elevation decreases (approximate p50/p95/p99 and the number of cells lowered by more than
1, 5, 10, 25 and 50 vertical units). `processing_summary.json` has `preflight`, `performance`, `whitebox_calls` (seconds and sampled
memory) and `stages_reused_from_previous_run`.

**Practical.** Run long jobs detached (`nohup ... &` or `tmux`) so a closed Jupyter session cannot kill them. Check
`cat /sys/fs/cgroup/memory.max`: the container limit may be below the machine's RAM. Point `outputs.directory` (and optionally
`execution.scratch_directory`) to local disk if the input lives on a network file system.

## 12. Tests

```bash
pytest                    # all tests
pytest -m "not wbt"       # skip tests that need the WhiteboxTools executable
```
Tests use synthetic rasters only (131 tests, including resume scenarios, stripe-independence of every block-wise stage and the GDAL rewrite). Tests marked `wbt` run the real executable (they are skipped if it is not found);
failure propagation is tested with a fake executable (`tests/fake_wbt.py`).
