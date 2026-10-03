# CIDI — Automated Floodplain Delineation from DEM

A reproducible Python workflow for generating a **potentially floodable floodplain map from a DEM** using a DEM-derived drainage network and Height Above Nearest Drainage (HAND).

> The resulting map represents a topographically defined **potential floodplain**, not observed inundation extent.

## Scope

This repository ends at the generation of the potential floodplain raster. It does not implement Hess-product comparison, scientific validation, classification, machine learning, ecological evaluation, hydrological validation, WSE/DAHITI integration, object-based classification, probability surfaces, hydraulic modelling, flood-frequency analysis, or satellite-observed inundation mapping.

## Inputs

The workflow requires:

- an arbitrary DEM raster;
- a pilot-area GeoPackage and layer;
- an explicitly configured input CRS;
- an explicitly configured projected processing/output CRS;
- a DEM-derived drainage-network scale expressed as a flow-accumulation threshold in cells;
- a processing buffer around the pilot area;
- a HAND threshold in metres.

No particular DEM product, spatial resolution, geographic extent, or external river network is hard-coded.

## CRS configuration

The YAML file distinguishes the original/input CRS from the processing/output CRS:

```yaml
crs:
  input: "EPSG:4326"
  output: "EPSG:31983"
```

If an input dataset already has CRS metadata, it must agree with `crs.input`; conflicting metadata are treated as an input error rather than silently transformed. If a source CRS is missing, the configured input CRS is used explicitly.

The processing CRS must be projected and use metres. This is required because drainage thresholds, buffers, raster resolution, and HAND are interpreted in metric processing space.

## Drainage-network scale

The drainage network is derived from the DEM with WhiteboxTools D8 flow accumulation and stream extraction. The configuration controls the minimum accumulation threshold:

```yaml
hydrology:
  stream_threshold:
    value: 1000
    units: "cells"
```

Lower thresholds generally produce denser derived drainage networks; higher thresholds generally produce sparser networks. Because HAND is referenced to this derived network, changing the threshold can materially change the floodplain result. The parameter is therefore exposed rather than hidden in the code.

## Processing buffer

Hydrological calculations are performed on a buffered DEM rather than a DEM clipped exactly to the pilot polygon. This reduces artificial flow-boundary effects at the pilot-area edge. The buffer is explicitly configured:

```yaml
hydrology:
  processing_buffer:
    value: 1000
    units: "m"
```

The example value is not a universal scientific recommendation. It should be selected for the study area and drainage-network scale.

## HAND threshold

The final map uses a simple configured HAND threshold:

```yaml
floodplain:
  hand_threshold:
    value: 5.0
    units: "m"
```

Cells satisfying `HAND <= threshold` are assigned value `1`; other cells are assigned value `0`. The example value is only an example configuration and is not asserted to be scientifically correct for every study area.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate floodplain-delineation
```

The environment uses the `conda-forge` channel and includes Rasterio, GeoPandas, NumPy, PyYAML, Shapely, PyProj, WhiteboxTools, and pytest.

## Run

Copy and edit the example configuration:

```bash
cp config/example.yaml config/my_project.yaml
```

Then run:

```bash
python scripts/run_pipeline.py --config config/my_project.yaml
```

The YAML file is the principal user interface. The CLI intentionally requires only the configuration path.

## Outputs

The default configuration produces:

```text
data/output/
├── conditioned_dem.tif
├── flow_accumulation.tif
├── flow_direction.tif
├── derived_drainage.tif
├── derived_drainage.gpkg
├── hand.tif
└── potential_floodplain.tif
```

### `conditioned_dem.tif`

DEM after WhiteboxTools depression filling, used for subsequent hydrological processing.

### `flow_direction.tif`

D8 flow-direction raster derived from the conditioned DEM.

### `flow_accumulation.tif`

D8 contributing-cell accumulation raster.

### `derived_drainage.tif`

Binary drainage raster extracted from flow accumulation using the configured stream threshold.

### `derived_drainage.gpkg`

Vector representation of the DEM-derived drainage network.

### `hand.tif`

Elevation above the nearest downslope derived drainage cell. This is retained as a key scientific intermediate.

### `potential_floodplain.tif`

Binary final product:

- `0` = outside potential floodplain;
- `1` = potential floodplain;
- `255` = NoData where applicable.

The raster is in the configured processing CRS, is masked to the pilot geometry, and contains metadata describing the method and HAND threshold.

## Tests

Run software tests with:

```bash
pytest
```

These tests cover configuration parsing, CRS handling, expected hydrological tool invocation, binary output generation, and NoData behavior. They are software tests, not scientific validation against an external floodplain product.

## Reproducibility considerations

The result depends strongly on:

- DEM quality;
- DEM resolution;
- source CRS and processing CRS;
- raster resolution and grid produced by reprojection;
- hydrological conditioning;
- drainage-network threshold;
- processing buffer;
- HAND threshold.

The workflow deliberately exposes these important assumptions through YAML rather than embedding them as hidden constants.

## Methodological references to implementation choices

WhiteboxTools provides the hydrological operations used here, including depression filling, D8 flow direction, D8 flow accumulation, stream extraction, raster-stream vectorization, and elevation above stream. The Python frontend is used as the automation interface.
