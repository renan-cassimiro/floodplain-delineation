"""Create a SYNTHETIC DEM and pilot area for trying the workflow.

The terrain is artificial (a gently sloping river valley with a flat floor, valley walls,
a tributary and smooth noise). It is only meant to exercise the pipeline; results say
nothing about any real floodplain.

Usage:  python scripts/make_synthetic_example.py --out-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


def build_dem(height: int = 240, width: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype("float64")
    # main valley runs west->east with a gentle meander; elevation falls toward the east
    center = height * 0.5 + 12 * np.sin(x / 38.0)
    dist = np.abs(y - center)
    floor_half_width = 14.0
    walls = np.clip(dist - floor_half_width, 0, None) * 0.55          # valley walls (m per cell)
    along = (width - x) * 0.012                                       # downstream slope
    # tributary entering from the north
    t_x = width * 0.45 + 6 * np.sin(y / 25.0)
    t_dist = np.abs(x - t_x)
    t_walls = np.clip(t_dist - 5, 0, None) * 0.45
    t_mask = y < center
    base = 30.0 + along + np.where(t_mask, np.minimum(walls, t_walls + (center - y) * 0.0), walls)
    # smooth noise (box-filtered white noise)
    noise = rng.normal(0, 0.35, size=(height, width))
    k = 5
    pad = np.pad(noise, k, mode="reflect")
    smooth = sum(pad[i : i + height, j : j + width] for i in range(2 * k + 1) for j in range(2 * k + 1)) / (2 * k + 1) ** 2
    return (base + smooth * 3.0).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dem = build_dem()
    nodata = -9999.0
    dem[:, :3] = nodata                      # nodata strip along the western edge
    res, x0, y0 = 30.0, 500000.0, 9000000.0  # 30 m cells, UTM 20S (EPSG:32720)
    transform = from_origin(x0, y0, res, res)
    with rasterio.open(
        out / "dem.tif", "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1,
        dtype="float32", crs="EPSG:32720", transform=transform, nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(dem, 1)

    # pilot area: central window leaving a margin for the context buffer
    h, w = dem.shape
    pilot = box(x0 + 60 * res, y0 - (h - 60) * res, x0 + (w - 60) * res, y0 - 60 * res)
    gpd.GeoDataFrame({"name": ["synthetic_pilot"]}, geometry=[pilot], crs="EPSG:32720").to_file(
        out / "pilot_area.gpkg", layer="pilot", driver="GPKG"
    )
    print(f"Wrote {out/'dem.tif'} and {out/'pilot_area.gpkg'} (synthetic data).")


if __name__ == "__main__":
    main()
