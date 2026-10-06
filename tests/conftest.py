"""Shared synthetic fixtures. No real datasets are used anywhere in the test suite."""

from __future__ import annotations

import copy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

NODATA = -9999.0
X0, Y0, RES = 500000.0, 9000000.0, 30.0
CRS = "EPSG:32720"


def write_dem(path: Path, array, *, crs=CRS, res=RES, x0=X0, y0=Y0, nodata=NODATA, dtype=None, transform=None):
    array = np.asarray(array)
    if dtype is not None:
        array = array.astype(dtype)
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
        dtype=array.dtype, crs=crs, transform=transform or from_origin(x0, y0, res, res), nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def write_gpkg(path: Path, geoms, *, crs=CRS, layer="pilot"):
    gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=list(geoms), crs=crs).to_file(path, layer=layer, driver="GPKG")
    return path


def valley_terrain(height=120, width=150, seed=1) -> np.ndarray:
    """Smooth synthetic valley draining west->east (float32), without nodata."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype("float64")
    center = height * 0.5 + 8 * np.sin(x / 30.0)
    walls = np.clip(np.abs(y - center) - 8, 0, None) * 0.6
    noise = rng.normal(0, 0.2, (height, width))
    k = 3
    pad = np.pad(noise, k, mode="reflect")
    smooth = sum(pad[i : i + height, j : j + width] for i in range(2 * k + 1) for j in range(2 * k + 1)) / (2 * k + 1) ** 2
    return (30.0 + (width - x) * 0.02 + walls + smooth * 3).astype("float32")


def config_dict(tmp_path: Path, dem: Path, gpkg: Path, **over):
    cfg = {
        "inputs": {"dem": str(dem), "pilot_area": str(gpkg), "pilot_layer": None, "vertical_units": "meters"},
        "processing": {
            "buffer_distance": 600,
            "buffer_units": "meters",
            "conditioning": {"method": "breach"},
            "flow": {"algorithm": "d8"},
            "drainage": {"threshold_type": "cells", "threshold_value": 150},
            "hand": {"threshold": 3},
        },
        "outputs": {"directory": str(tmp_path / "out")},
        "execution": {"overwrite": False, "verbose": False},
    }
    for dotted, value in over.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return cfg


@pytest.fixture
def base_cfg(tmp_path):
    """A valid configuration dict pointing at (not necessarily existing) files."""
    return copy.deepcopy(config_dict(tmp_path, tmp_path / "dem.tif", tmp_path / "pilot.gpkg"))


@pytest.fixture
def valley_dataset(tmp_path):
    """DEM + pilot GPKG on the same CRS; pilot leaves a margin for the buffer."""
    arr = valley_terrain()
    dem = write_dem(tmp_path / "dem.tif", arr)
    h, w = arr.shape
    pilot = box(X0 + 30 * RES, Y0 - (h - 30) * RES, X0 + (w - 30) * RES, Y0 - 30 * RES)
    gpkg = write_gpkg(tmp_path / "pilot.gpkg", [pilot])
    return dem, gpkg, arr


def wbt_executable_available() -> bool:
    try:
        from floodplain_delineation.wbt import locate_executable

        locate_executable()
        return True
    except Exception:
        return False


requires_wbt = pytest.mark.skipif(not wbt_executable_available(), reason="WhiteboxTools executable not available")
