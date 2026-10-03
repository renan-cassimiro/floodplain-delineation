# Methodology

## Scope

This version implements only a reproducible DEM-derived potential floodplain workflow. It does not compare the result with external floodplain products and does not perform scientific validation, classification, machine learning, hydraulic modelling, ecological evaluation, or observational inundation mapping.

## Processing logic

1. Read the pilot-area geometry from the configured GeoPackage layer.
2. Validate the configured input CRS against source metadata. A missing source CRS is assigned the configured input CRS; a conflicting source CRS raises an error.
3. Crop the DEM to a user-configured buffer around the pilot area.
4. Reproject the buffered DEM into the configured projected processing CRS.
5. Fill depressions with WhiteboxTools.
6. Calculate D8 flow direction and D8 flow accumulation.
7. Extract drainage cells using the configured flow-accumulation threshold.
8. Convert the drainage raster to a vector GeoPackage.
9. Calculate elevation above the nearest downslope stream cell using WhiteboxTools `elevation_above_stream`.
10. Classify cells with HAND less than or equal to the configured threshold as potential floodplain.
11. Mask the final binary raster to the pilot-area geometry.

## Interpretation

The resulting map represents a topographically defined **potential floodplain**, not observed inundation extent.

The result is sensitive to DEM quality and resolution, hydrological conditioning, the derived drainage-network scale, the HAND threshold, the processing CRS, and the raster grid produced during reprojection.
