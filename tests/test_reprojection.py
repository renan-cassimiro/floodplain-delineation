import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS

from floodplain_delineation.reprojection import reproject_dem


def test_reproject_dem_creates_projected_grid_and_reuses_tagged_output(tmp_path):
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "projected.tif"
    data = np.arange(100, dtype="float32").reshape(10, 10)
    with rasterio.open(src_path, "w", driver="GTiff", height=10, width=10, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(-50, -5, 0.01, 0.01),
                       nodata=-9999) as dst:
        dst.write(data, 1)
    result = reproject_dem(src_path, dst_path, "ESRI:102033", None, "bilinear")
    with rasterio.open(dst_path) as out:
        assert CRS.from_user_input(out.crs).is_projected
        assert out.width > 1 and out.height > 1
        assert out.crs.to_string().upper().startswith("ESRI")
    assert result["reused"] is False
    cached = reproject_dem(src_path, dst_path, "ESRI:102033", None, "bilinear", resume=True)
    assert cached["reused"] is True
