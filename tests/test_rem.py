import numpy as np
import rasterio
from rasterio.transform import from_origin

from floodplain_delineation.rem import _idw_block, _sample_points


def test_idw_returns_exact_sample_elevation_at_sample_location():
    samples = np.array([[0.0, 0.0, 12.5], [10.0, 0.0, 30.0]])
    result = _idw_block(np.array([[0.0, 0.0]]), samples, power=2.0)
    assert result[0] == np.float32(12.5)


def test_idw_uses_all_samples_and_euclidean_distance():
    samples = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 20.0]])
    result = _idw_block(np.array([[5.0, 0.0]]), samples, power=2.0)
    assert np.isclose(result[0], 15.0)


def test_sampling_is_systematic_and_separate_per_basin(tmp_path):
    dem_path = tmp_path / "dem.tif"
    drain_path = tmp_path / "drain.tif"
    basin_path = tmp_path / "basins.tif"
    transform = from_origin(0, 4, 1, 1)
    profile = dict(driver="GTiff", height=4, width=4, count=1, crs="EPSG:3857",
                   transform=transform, nodata=-9999)
    dem = np.arange(16, dtype="float32").reshape(4, 4)
    drainage = np.ones((4, 4), dtype="uint8")
    basins = np.array([[1, 1, 2, 2]] * 4, dtype="int32")
    for path, arr, dtype, nodata in ((dem_path, dem, "float32", -9999), (drain_path, drainage, "uint8", 0),
                                     (basin_path, basins, "int32", 0)):
        with rasterio.open(path, "w", **{**profile, "dtype": dtype, "nodata": nodata}) as dst:
            dst.write(arr, 1)
    samples = _sample_points(dem_path, drain_path, basin_path, interval=2)
    assert set(samples) == {1, 2}
    assert len(samples[1]) == 4
    assert len(samples[2]) == 4


def test_compute_rem_interpolates_independently_within_each_basin(tmp_path):
    from floodplain_delineation.logging_utils import WarningCollector
    from floodplain_delineation.rem import compute_rem

    dem_path = tmp_path / "conditioned.tif"
    drain_path = tmp_path / "drain.tif"
    pointer_path = tmp_path / "pointer.tif"
    basin_path = tmp_path / "basins.tif"
    surface_path = tmp_path / "surface.tif"
    rem_path = tmp_path / "rem.tif"
    transform = from_origin(0, 4, 1, 1)
    profile = dict(driver="GTiff", height=4, width=4, count=1, crs="EPSG:3857",
                   transform=transform, nodata=-9999)
    dem = np.array([[10, 10, 100, 100], [10, 10, 100, 100],
                    [10, 10, 100, 100], [10, 10, 100, 100]], dtype="float32")
    drainage = np.zeros((4, 4), dtype="uint8")
    drainage[0, 0] = drainage[0, 3] = 1
    for path, arr, dtype in ((dem_path, dem, "float32"), (drain_path, drainage, "uint8"),
                             (pointer_path, np.ones((4, 4), dtype="uint8"), "uint8")):
        with rasterio.open(path, "w", **{**profile, "dtype": dtype,
                                          "nodata": -9999 if dtype == "float32" else 255}) as dst:
            dst.write(arr, 1)
    basin_ids = np.array([[1, 1, 2, 2]] * 4, dtype="int32")

    class StubRunner:
        def run_raster(self, tool, flags, *, work_dir, output):
            assert tool == "Basins"
            with rasterio.open(output, "w", **{**profile, "dtype": "int32", "nodata": 0}) as dst:
                dst.write(basin_ids, 1)

    result = compute_rem(StubRunner(), dem_path, drain_path, pointer_path, basin_path,
                         surface_path, rem_path, tmp_path, 1, 2.0, WarningCollector())
    with rasterio.open(surface_path) as src:
        surface = src.read(1)
    with rasterio.open(rem_path) as src:
        rem = src.read(1)
    assert result["basins"] == 2
    assert np.allclose(surface[:, :2], 10.0)
    assert np.allclose(surface[:, 2:], 100.0)
    assert np.allclose(rem, 0.0)
