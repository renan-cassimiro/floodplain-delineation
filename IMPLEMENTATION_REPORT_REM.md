# Implementation report — basin-wise IDW REM and global processing CRS

## Implemented

- Added an initial DEM reprojection stage to the configured projected processing CRS. Default: `ESRI:102033` (South America Albers Equal Area Conic). It is deliberately not labelled `EPSG:102033`.
- Added optional target resolution and DEM resampling (`nearest`, `bilinear`, `cubic`; default `bilinear`). When target resolution is `null`, GDAL's suggested target resolution is used.
- The reprojected DEM is retained at `intermediate/dem_processing_crs.tif` and becomes the input to spatial preparation, conditioning, D8, accumulation, drainage extraction, HAND and REM.
- Added optional REM configuration (`enabled`, `sampling_interval_pixels`, `idw_power`). Default is disabled, preserving HAND-only runs unless explicitly enabled.
- Added WhiteboxTools `Basins` from the existing D8 pointer raster, using the existing `WhiteboxRunner` and its version/help/normalization mechanisms.
- Added systematic sampling of every Nth drainage pixel per basin in deterministic raster scan order. Elevations are taken from the conditioned DEM.
- Added exact all-sample Euclidean IDW independently per basin, with bounded temporary pairwise-distance arrays, followed by `REM = conditioned DEM - interpolated drainage-elevation surface`.
- Added basin IDs, interpolated surface and REM rasters; added YAML examples and documentation.
- Bumped the stage-manifest schema to prevent reuse of outputs produced under the former source-CRS processing semantics.

## Configuration

```yaml
processing:
  crs: "ESRI:102033"
  target_resolution: null
  dem_resampling: "bilinear"
  rem:
    enabled: true
    sampling_interval_pixels: 10
    idw_power: 2.0
```

## Validation performed

- Python compilation: passed.
- Focused REM and reprojection tests: passed (5 tests).
- Configuration, pipeline, Whitebox wrapper and new-feature tests: 59 passed, 7 skipped.
- Blockwise, delineation, hydrology, input, resume and runner tests: 54 passed, 15 skipped.
- Across the two test groups, 113 tests passed and 22 were skipped. Skips are the repository's tests requiring a real WhiteboxTools executable.
- Both example YAML files load successfully with the new configuration.

## Not validated in this environment

- No WhiteboxTools executable is installed in this execution environment. Therefore the actual `Basins` tool help/flag signature and its output semantics were not exercised here. The implementation asks the installed runner to verify that `Basins` exists and verifies its output grid and integer basin IDs at runtime. Run a small synthetic integration case with the same WhiteboxTools version used in the target environment before processing the Amazon basin.
- No full-size DEM was processed. Exact all-sample IDW has computational complexity proportional to target cells multiplied by the number of samples in each basin. Temporary distance arrays are chunked to limit memory, but runtime can be very high for large basins and dense sampling. No search radius or neighbor cap was added because that would change the specified method.
- The DEM is reprojected over its full source extent before the pilot-area window is extracted. Ensure sufficient disk space for this additional raster.

## Files changed/added

- `README.md`
- `config/amazon_basin.yaml`
- `config/example.yaml`
- `docs/methodology.md`
- `src/floodplain_delineation/config.py`
- `src/floodplain_delineation/inputs.py`
- `src/floodplain_delineation/manifest.py`
- `src/floodplain_delineation/outputs.py`
- `src/floodplain_delineation/pipeline.py`
- `src/floodplain_delineation/preprocessing.py`
- `src/floodplain_delineation/rem.py` (new)
- `src/floodplain_delineation/reprojection.py` (new)
- `src/floodplain_delineation/wbt.py`
- `tests/test_rem.py` (new)
- `tests/test_reprojection.py` (new)
