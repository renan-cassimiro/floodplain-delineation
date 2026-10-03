"""End-to-end floodplain delineation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling

from .config import Config
from .crs import validate_processing_crs
from .dem import prepare_dem
from .exceptions import InputError
from .floodplain import create_floodplain
from .hand import calculate_hand
from .hydrology import convert_drainage_to_geopackage, run_hydrology
from .io import read_pilot_area, validate_dem

LOGGER = logging.getLogger(__name__)


def _reproject_pilot(pilot_area: gpd.GeoDataFrame, output_crs) -> gpd.GeoDataFrame:
    return pilot_area.to_crs(output_crs)


def _reproject_cropped_dem_to_processing_crs(source_path: Path, destination_path: Path, output_crs) -> None:
    """Reproject the buffered DEM crop into the processing CRS."""
    with rasterio.open(source_path) as src:
        source_crs = src.crs or output_crs
        transform, width, height = calculate_default_transform(
            source_crs,
            output_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(
            crs=output_crs,
            transform=transform,
            width=width,
            height=height,
            compress="deflate",
            predictor=3,
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=source_crs,
                dst_transform=transform,
                dst_crs=output_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )


def run_pipeline(config: Config) -> dict[str, Path]:
    """Execute the complete DEM -> drainage -> HAND -> floodplain workflow."""
    validate_processing_crs(config.output_crs)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading configuration...")
    LOGGER.info("Reading pilot area...")
    pilot = read_pilot_area(config.pilot_area, config.pilot_layer, config.input_crs)
    validate_dem(config.dem, config.input_crs)

    LOGGER.info("Reading DEM...")
    LOGGER.info("Checking CRS...")
    LOGGER.info("Preparing DEM...")
    # The first crop is performed in the input CRS, preserving the explicit input/output distinction.
    cropped_input = config.output_dir / "_working_buffered_dem_input_crs.tif"
    prepare_dem(
        config.dem,
        pilot,
        cropped_input,
        config.input_crs,
        config.processing_buffer.value,
    )

    LOGGER.info("Reprojecting data...")
    processing_dem = config.output_dir / "_working_buffered_dem_processing_crs.tif"
    _reproject_cropped_dem_to_processing_crs(
        cropped_input,
        processing_dem,
        config.output_crs,
    )

    LOGGER.info("Deriving flow accumulation...")
    run_hydrology(
        processing_dem,
        config.outputs["conditioned_dem"],
        config.outputs["flow_direction"],
        config.outputs["flow_accumulation"],
        config.outputs["drainage_raster"],
        int(config.stream_threshold.value),
        config.output_dir / "_working_drainage.shp",
    )

    LOGGER.info("Extracting drainage network...")
    convert_drainage_to_geopackage(
        config.output_dir / "_working_drainage.shp",
        config.outputs["drainage"],
        config.output_crs,
    )

    LOGGER.info("Calculating HAND...")
    calculate_hand(
        config.outputs["conditioned_dem"],
        config.outputs["drainage_raster"],
        config.outputs["hand"],
    )

    LOGGER.info("Applying floodplain threshold...")
    create_floodplain(
        config.outputs["hand"],
        _reproject_pilot(pilot, config.output_crs),
        config.outputs["floodplain"],
        config.hand_threshold.value,
    )

    # Remove temporary working files while retaining scientific intermediate products.
    for path in [cropped_input, processing_dem]:
        path.unlink(missing_ok=True)
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        (config.output_dir / f"_working_drainage{suffix}").unlink(missing_ok=True)

    LOGGER.info("Writing outputs...")
    LOGGER.info("Pipeline completed.")
    return config.outputs.copy()
