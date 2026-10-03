"""DEM conditioning, D8 drainage derivation, and stream vectorization."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio

from .exceptions import ProcessingError
from .io import write_geodataframe


class WhiteboxRunner:
    """Small adapter around the WhiteboxTools Python frontend."""

    def __init__(self, work_dir: Path):
        try:
            from whitebox import WhiteboxTools
        except ImportError as exc:
            raise ProcessingError(
                "WhiteboxTools Python package is unavailable. Create the Conda environment first."
            ) from exc
        self.wbt = WhiteboxTools()
        self.wbt.work_dir = str(work_dir)
        self.wbt.verbose = False

    def run(self, tool: str, args: list[str]) -> None:
        result = self.wbt.run_tool(tool, args)
        if result not in (0, None):
            raise ProcessingError(f"WhiteboxTools operation '{tool}' failed with code {result}.")


def _arg(name: str, value: Path | str) -> str:
    return f"{name}={value}"


def run_hydrology(
    dem_path: Path,
    conditioned_dem: Path,
    flow_direction: Path,
    flow_accumulation: Path,
    drainage_raster: Path,
    stream_threshold_cells: int,
    drainage_vector_tmp: Path,
) -> None:
    """Run WhiteboxTools hydrological preprocessing and stream extraction."""
    runner = WhiteboxRunner(dem_path.parent)
    runner.run(
        "fill_depressions",
        [
            _arg("--dem", dem_path),
            _arg("--output", conditioned_dem),
        ],
    )
    runner.run(
        "d8_pointer",
        [
            _arg("--dem", conditioned_dem),
            _arg("--output", flow_direction),
        ],
    )
    runner.run(
        "d8_flow_accumulation",
        [
            _arg("--input", flow_direction),
            _arg("--output", flow_accumulation),
            "--pntr",
            "--out_type=cells",
        ],
    )
    runner.run(
        "extract_streams",
        [
            _arg("--flow_accum", flow_accumulation),
            _arg("--output", drainage_raster),
            f"--threshold={stream_threshold_cells}",
            "--zero_background",
        ],
    )
    runner.run(
        "raster_streams_to_vector",
        [
            _arg("--streams", drainage_raster),
            _arg("--output", drainage_vector_tmp),
        ],
    )


def convert_drainage_to_geopackage(
    temporary_vector: Path,
    output_gpkg: Path,
    output_crs,
) -> None:
    """Convert the Whitebox stream vector to the requested GeoPackage output."""
    if not temporary_vector.exists():
        raise ProcessingError(f"Expected drainage vector was not created: {temporary_vector}")
    gdf = gpd.read_file(temporary_vector)
    if gdf.empty:
        raise ProcessingError("Derived drainage network contains no stream features.")
    if gdf.crs is None:
        gdf = gdf.set_crs(output_crs)
    else:
        gdf = gdf.to_crs(output_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ProcessingError("Derived drainage network contains no valid geometries.")
    write_geodataframe(gdf, output_gpkg, layer="drainage")
