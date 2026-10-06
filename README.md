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

* Only the **pilot geometry** is reprojected (to the DEM CRS). The DEM is never reprojected or resampled.
* The pilot area must intersect the DEM footprint, otherwise the run fails with a clear error.
* Integer DEMs are stored as float32 in the intermediate extract (values unchanged). A NaN nodata is
  replaced by the explicit sentinel `-9999` (warning recorded).
* Prefer a **projected CRS** with metric units. Area-based thresholds on geographic CRSs use the geodesic
  cell area at the extent's central latitude and emit a warning quantifying the latitude variation.

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
| `inputs.dem`, `inputs.pilot_area` | yes | paths | |
| `inputs.pilot_layer` | no | layer name or `null` | warning if the GPKG has several layers and none is named |
| `inputs.vertical_units` | yes | `meters`, `centimeters`, `decimeters`, `feet`, `us_survey_feet` | documents the unit of DEM and HAND threshold |
| `processing.buffer_distance` | yes | `>= 0`, in `buffer_units` | context around the pilot area |
| `processing.buffer_units` | no (default `meters`) | `meters`, `cells` | geographic DEMs: metric buffer computed in a local UTM CRS and transformed back |
| `processing.conditioning.method` | yes | `breach`, `fill` | see section 6 |
| `processing.flow.algorithm` | yes | `d8` | the only supported algorithm |
| `processing.drainage.threshold_type` | yes | `cells`, `area_km2` | |
| `processing.drainage.threshold_value` | yes | `> 0` (whole number for `cells`) | minimum contributing cells / km2 |
| `processing.hand.threshold` | yes | `>= 0`, DEM vertical units | |
| `outputs.directory` | yes | path | |
| `outputs.intermediates`, `outputs.final` | no (`intermediate`, `final`) | relative names | must differ, no `..` |
| `outputs.save_*` | no (`true`) | bool | `false` deletes that intermediate after the run |
| `execution.overwrite` | no (`false`) | bool | conflicts are checked **before** any processing |
| `execution.verbose` | no (`true`) | bool | DEBUG vs INFO logging |
| `execution.whitebox_executable` | no | path | |

## 6. Processing methodology

1. **Spatial preparation.** Pilot geometry is validated/repaired, reprojected, dissolved, intersected with the
   DEM footprint, buffered, and converted to an **integer pixel window of the original DEM grid** (so no
   resampling). Sides where the DEM edge truncates the requested buffer are reported. *A buffer cannot guarantee
   correct upstream hydrology if the DEM does not extend far enough beyond the area of interest.*
2. **Conditioning** (`dem_processing_extent.tif` is the unconditioned copy; the source DEM is never modified).
   * `breach` (**default**) -> `BreachDepressions` (Lindsay 2016). Lowers terrain along least-cost paths; keeps
     most elevations unchanged; recommended by WhiteboxTools over filling. The number of modified cells and the
     maximum elevation change are reported.
   * `fill` -> `FillDepressionsWangAndLiu --fix_flats`. Raises terrain inside depressions, which can create
     artificial flats. See *Known issues*.
   The result is validated: same grid, finite values, and an **identical nodata footprint**; any difference is an error.
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

## 7. Outputs

```
outputs/
├── intermediate/
│   ├── dem_processing_extent.tif   unconditioned DEM over the processing extent (source dtype, float32 for ints)
│   ├── conditioned_dem.tif         conditioned DEM (processing extent)
│   ├── flow_direction.tif          D8 codes: 1=NE 2=E 4=SE 8=S 16=SW 32=W 64=NW 128=N, 0=no downstream neighbour
│   ├── flow_accumulation.tif       contributing cells
│   ├── drainage_network.tif        uint8: 1 drainage, 0 other, 255 nodata
│   └── hand.tif                    HAND in DEM vertical units (nodata where no drainage association)
├── final/
│   └── potential_floodplain.tif    uint8: 1 potential floodplain, 0 other, 255 nodata; clipped to the pilot area
├── processing_area.gpkg            pilot (clipped / full), requested and effective processing extents
├── processing_summary.json         parameters, CRS, resolution, thresholds, stats, versions, warnings, limitations
└── processing.log
```

All rasters keep the DEM CRS and resolution; the final mask stays pixel-aligned with the DEM grid. Intermediates cover the
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

## 9. Known issues observed with the installed WhiteboxTools (v2.0.0 binary, tested here)

* `FillDepressions` terminated with exit status 101 (`Error unwrapping 'output'`) on 3 of 5 synthetic DEMs, and worked on 2;
  the cause was not identified. It is therefore **not used**; `fill` uses `FillDepressionsWangAndLiu`.
* On a DEM with an *interior* nodata hole, `FillDepressionsWangAndLiu` left an integer sentinel (-2147483648) as valid data in
  some hole cells and `FillDepressionsPlanchonAndDarboux` filled the hole with invented elevations. The pipeline's nodata-footprint
  check turns this into an explicit error. `breach` did not show the problem. These observations come from small synthetic tests
  and are not a general statement about WhiteboxTools.
* WhiteboxTools ignores unknown flags silently and `whitebox.run_tool` returns 0 even if the tool panics. This project calls the
  binary directly, checks the exit status, checks every flag against the installed tool's help text, and verifies outputs exist.

## 10. Reproducibility

The summary stores the effective configuration, the SHA-256 of the YAML file, input paths, software versions (Python, numpy,
rasterio, GDAL, geopandas, shapely, pyproj, PyYAML, `whitebox` package, WhiteboxTools binary), every WhiteboxTools call (tool,
flags, exit status), UTC timestamps, warnings and limitations. Output organisation is fixed as shown above.
Reusing `hand.tif` to try another HAND threshold is possible manually; no caching system is implemented.

## 11. Tests

```bash
pytest                    # all tests
pytest -m "not wbt"       # skip tests that need the WhiteboxTools executable
```
Tests use synthetic rasters only. Tests marked `wbt` run the real executable (they are skipped if it is not found);
failure propagation is tested with a fake executable (`tests/fake_wbt.py`).
