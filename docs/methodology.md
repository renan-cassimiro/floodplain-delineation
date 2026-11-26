# Methodology

## Scope

This version implements a reproducible, DEM-derived workflow with two products: a **potential floodplain** mask (HAND) and,
optionally, a **Relative Elevation Model** (REM). It does not compare results with external floodplain products and does not
perform scientific validation, classification, machine learning, hydraulic modelling, ecological evaluation or observational
inundation mapping.

## Processing logic

0. **Processing grid and DEM reprojection.** The processing CRS is projected (default `ESRI:102033`, South America Albers Equal
   Area Conic; ESRI authority, never an EPSG code). The extent (pilot area + buffer, intersected with the DEM footprint
   transformed with densified edges) defines a grid whose resolution is configured or derived by GDAL and whose origin is
   snapped to a multiple of the resolution. The DEM is warped onto it in fixed-height stripes. A cell is valid only if the
   nearest source cell is valid (nodata is never interpolated across). A DEM already in the processing CRS with no requested
   resolution is only cropped.
1. **Conditioning** (on the unconditioned DEM of stage 0): `breach`, `fill`, or `breach_least_cost`
   (`BreachDepressionsLeastCost` with configurable maximum path length, maximum cost, distance minimisation and flat increment),
   optionally followed by `FillDepressionsWangAndLiu --fix_flats` for the depressions the bounded breach could not cross.
2. **D8 flow direction and accumulation** (WhiteboxTools).
3. **Drainage network**: cells with accumulation >= the configured threshold (cells or km2 of the processing CRS).
4. **HAND**: elevation above the nearest downslope drainage cell along the D8 path (conditioned DEM).
5. **Potential floodplain**: valid cells with HAND <= the configured threshold, clipped to the pilot area.
6. **REM** (optional, independent of HAND):
   1. *Basins*: WhiteboxTools `Subbasins` gives the exclusive, non-nested catchment of each stream link.
   2. *Sampling*: per N x N block and basin, the drainage pixel closest to the block centre; elevation from the
      unconditioned DEM of stage 0.
   3. *Interpolation*: IDW with all samples of the cell's own basin, euclidean distances in the processing CRS, exponent `p`.
   4. *REM*: DEM minus the surface; negative values are kept.

## Interpretation

The HAND mask is a topographically defined **potential floodplain**, not observed inundation extent. The REM is a relative
elevation product above a surface interpolated from sampled drainage elevations; it is not a water surface and is not validated.

Both are sensitive to DEM quality and resolution, the resampling to the processing grid, the conditioning, the drainage
threshold, the processing CRS and, for the REM, the basin units, the sampling interval and the IDW exponent. HAND and the REM
use different reference elevations (conditioned vs unconditioned DEM) and different association rules (D8 path vs per-basin IDW
surface), so they are not expected to agree cell by cell.
