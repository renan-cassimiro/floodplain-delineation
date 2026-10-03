import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box

from floodplain_delineation.floodplain import create_floodplain


def test_binary_output_and_nodata_propagation(tmp_path):
    hand_path = tmp_path / "hand.tif"
    output_path = tmp_path / "floodplain.tif"
    transform = from_origin(0, 4, 1, 1)
    data = np.array([[1, 4, 6, -9999], [2, 8, 3, 4], [9, 1, 7, 2], [1, 1, 1, 1]], dtype=np.float32)
    with rasterio.open(
        hand_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:31983",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)

    pilot = gpd.GeoDataFrame(
        geometry=[box(0, 0, 4, 4)],
        crs="EPSG:31983",
    )
    create_floodplain(hand_path, pilot, output_path, 5)

    with rasterio.open(output_path) as src:
        result = src.read(1)
        assert src.nodata == 255
        assert set(np.unique(result)).issubset({0, 1})
        assert result[0, 0] == 1
        assert result[0, 2] == 0
