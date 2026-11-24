# Methodology

## Scope

This version implements only a reproducible DEM-derived potential floodplain workflow. It does not compare the result with external floodplain products and does not perform scientific validation, classification, machine learning, hydraulic modelling, ecological evaluation, or observational inundation mapping.

## Processing logic

1. Validate and reproject the full source DEM to the configured projected processing CRS and master grid.
2. Read, validate and reproject the pilot-area geometry to the processing CRS.
3. Derive the buffered processing window on the reprojected DEM grid.
4. Condition the DEM with the configured WhiteboxTools method.
5. Calculate D8 flow direction and D8 flow accumulation.
6. Extract drainage cells using the configured flow-accumulation threshold.
7. Calculate HAND with WhiteboxTools `ElevationAboveStream`.
8. Optionally, delineate basin IDs with WhiteboxTools `Basins` from the existing D8 pointer, sample drainage elevations and calculate basin-wise IDW REM.
9. Classify cells with HAND less than or equal to the configured threshold as potential floodplain.
10. Mask the final binary raster to the pilot-area geometry.

## Interpretation

The resulting map represents a topographically defined **potential floodplain**, not observed inundation extent.

The result is sensitive to DEM quality and resolution, hydrological conditioning, the derived drainage-network scale, the HAND threshold, the processing CRS, and the raster grid produced during reprojection.

## Processing CRS and basin-wise REM

The input DEM is first reprojected to the configured processing CRS (default `ESRI:102033`,
South America Albers Equal Area Conic). The DEM is then processed on a single master grid;
conditioning, D8 routing, accumulation, drainage extraction, HAND, basin delineation and REM
all use that grid. The output resolution can be configured; if omitted, GDAL's suggested
resolution is used. Reprojection and resampling may change elevation values and therefore
change downstream hydrological results compared with processing in the source CRS.

The optional REM uses WhiteboxTools `Basins` on the existing D8 pointer raster. Elevation
samples are taken from drainage cells in the conditioned DEM using a user-configured systematic
pixel interval. For each basin independently, all its samples contribute to Euclidean IDW:

`Z_IDW(x) = sum(z_i / d_i^p) / sum(1 / d_i^p)`

The REM is `conditioned_DEM - Z_IDW`. No samples from another basin contribute to a cell's
estimate. The resulting reference surface is topographic, because sample elevations originate
from the conditioned DEM; it is not a measured water surface or hydraulic simulation. Exact
all-sample IDW is computationally expensive for large basins and dense sampling.
