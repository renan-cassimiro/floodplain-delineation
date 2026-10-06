import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import Affine, from_origin
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform as shp_transform

from conftest import CRS, NODATA, RES, X0, Y0, config_dict, write_dem, write_gpkg
from floodplain_delineation.config import parse_config
from floodplain_delineation.exceptions import InputError
from floodplain_delineation.inputs import (
    prepare_spatial_domain, read_dem_info, read_pilot_geometry, validate_input_files,
)
from floodplain_delineation.logging_utils import WarningCollector
from floodplain_delineation.preprocessing import extract_processing_dem
from floodplain_delineation.spatial import cell_geometry, to_pyproj


def cfg_for(tmp_path, dem, gpkg, **over):
    return parse_config(config_dict(tmp_path, dem, gpkg, **over))


def pilot_box(col0, row0, col1, row1):
    return box(X0 + col0 * RES, Y0 - row1 * RES, X0 + col1 * RES, Y0 - row0 * RES)


# ----------------------------- files & DEM metadata ----------------------------- #
def test_missing_input_files(tmp_path):
    dem = write_dem(tmp_path / "dem.tif", np.ones((10, 10), "float32"))
    gpkg = write_gpkg(tmp_path / "p.gpkg", [pilot_box(1, 1, 5, 5)])
    validate_input_files(cfg_for(tmp_path, dem, gpkg))  # all present -> OK
    with pytest.raises(InputError, match="DEM file not found"):
        validate_input_files(cfg_for(tmp_path, tmp_path / "missing.tif", gpkg))
    with pytest.raises(InputError, match="Pilot-area GeoPackage file not found"):
        validate_input_files(cfg_for(tmp_path, dem, tmp_path / "missing.gpkg"))


def test_dem_without_crs_is_rejected(tmp_path):
    p = tmp_path / "nocrs.tif"
    with rasterio.open(p, "w", driver="GTiff", height=5, width=5, count=1, dtype="float32",
                       transform=from_origin(0, 5, 1, 1), nodata=-9999) as d:
        d.write(np.ones((5, 5), "float32"), 1)
    with pytest.raises(InputError, match="no coordinate reference system"):
        read_dem_info(p)


def test_dem_without_nodata_is_rejected(tmp_path):
    p = write_dem(tmp_path / "nonodata.tif", np.ones((5, 5), "float32"), nodata=None)
    with pytest.raises(InputError, match="no nodata"):
        read_dem_info(p)


def test_multiband_and_rotated_dems_are_rejected(tmp_path):
    p = tmp_path / "two.tif"
    with rasterio.open(p, "w", driver="GTiff", height=5, width=5, count=2, dtype="float32",
                       crs=CRS, transform=from_origin(0, 5, 1, 1), nodata=-1) as d:
        d.write(np.ones((2, 5, 5), "float32"))
    with pytest.raises(InputError, match="single-band"):
        read_dem_info(p)
    rot = write_dem(tmp_path / "rot.tif", np.ones((5, 5), "float32"), transform=Affine(1, 0.2, 0, 0.2, -1, 5))
    with pytest.raises(InputError, match="north-up"):
        read_dem_info(rot)


def test_unreadable_dem(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_text("not a raster")
    with pytest.raises(InputError, match="Cannot open DEM"):
        read_dem_info(bad)


# ------------------------------- pilot geometry ------------------------------- #
@pytest.fixture
def dem_info(tmp_path):
    return read_dem_info(write_dem(tmp_path / "d.tif", np.ones((100, 100), "float32")))


def test_empty_pilot_layer(tmp_path, dem_info):
    g = write_gpkg(tmp_path / "e.gpkg", [])
    with pytest.raises(InputError, match="no features"):
        read_pilot_geometry(g, None, dem_info, WarningCollector())


def test_non_polygon_pilot_layer(tmp_path, dem_info):
    g = write_gpkg(tmp_path / "pts.gpkg", [Point(X0 + 10, Y0 - 10)])
    with pytest.raises(InputError, match="no valid polygon"):
        read_pilot_geometry(g, None, dem_info, WarningCollector())


def test_pilot_without_crs(tmp_path, dem_info):
    import geopandas as gpd

    g = tmp_path / "nocrs.gpkg"
    gpd.GeoDataFrame(geometry=[pilot_box(1, 1, 5, 5)]).to_file(g, driver="GPKG")
    with pytest.raises(InputError, match="no CRS"):
        read_pilot_geometry(g, None, dem_info, WarningCollector())


def test_invalid_polygon_is_repaired_with_warning(tmp_path, dem_info):
    bowtie = Polygon([(X0 + 300, Y0 - 300), (X0 + 900, Y0 - 900), (X0 + 900, Y0 - 300), (X0 + 300, Y0 - 900)])
    assert not bowtie.is_valid
    g = write_gpkg(tmp_path / "bow.gpkg", [bowtie])
    w = WarningCollector()
    geom, _, _ = read_pilot_geometry(g, None, dem_info, w)
    assert geom.is_valid and geom.area > 0
    assert any("Repaired 1 invalid" in m for m in w.messages)


def test_multiple_polygons_are_dissolved_and_layer_selection(tmp_path, dem_info):
    a, b = pilot_box(5, 5, 20, 20), pilot_box(15, 15, 40, 40)   # overlapping
    g = write_gpkg(tmp_path / "m.gpkg", [a, b], layer="L1")
    geom, _, n = read_pilot_geometry(g, "L1", dem_info, WarningCollector())
    assert n == 2 and geom.geom_type == "Polygon"
    assert geom.area == pytest.approx(a.union(b).area)
    with pytest.raises(InputError, match="Layer 'nope' not found"):
        read_pilot_geometry(g, "nope", dem_info, WarningCollector())


def test_pilot_crs_transformation_to_dem_crs(tmp_path, dem_info):
    to_ll = pyproj.Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
    original = pilot_box(10, 10, 40, 40)
    ll = shp_transform(to_ll, original)
    g = write_gpkg(tmp_path / "ll.gpkg", [ll], crs="EPSG:4326")
    geom, src, _ = read_pilot_geometry(g, None, dem_info, WarningCollector())
    assert "4326" in src
    # round trip within a few centimetres
    assert geom.bounds == pytest.approx(original.bounds, abs=0.1)


def test_empty_intersection_is_an_error(tmp_path):
    dem = write_dem(tmp_path / "d.tif", np.ones((50, 50), "float32"))
    far = box(X0 + 1e6, Y0 - 1e6, X0 + 1e6 + 500, Y0 - 1e6 + 500)
    g = write_gpkg(tmp_path / "far.gpkg", [far])
    cfg = cfg_for(tmp_path, dem, g)
    with pytest.raises(InputError, match="does not intersect"):
        prepare_spatial_domain(read_dem_info(dem), cfg, WarningCollector())


# ------------------------------ buffer & window ------------------------------- #
def test_window_is_grid_aligned_and_includes_buffer(tmp_path):
    dem = write_dem(tmp_path / "d.tif", np.ones((200, 200), "float32"))
    pilot = pilot_box(60.3, 60.3, 120.6, 120.6)  # not on the grid
    g = write_gpkg(tmp_path / "p.gpkg", [pilot])
    cfg = cfg_for(tmp_path, dem, g, **{"processing.buffer_distance": 450})
    info = read_dem_info(dem)
    w = WarningCollector()
    d = prepare_spatial_domain(info, cfg, w)
    l, b, r, t = d.extent_bounds
    # covers pilot + 450 m (=15 cells)
    assert l <= pilot.bounds[0] - 450 + 1e-6 and r >= pilot.bounds[2] + 450 - 1e-6
    assert b <= pilot.bounds[1] - 450 + 1e-6 and t >= pilot.bounds[3] + 450 - 1e-6
    # aligned to the DEM grid (integer number of cells from DEM origin) -> no resampling
    assert ((l - X0) / RES) == pytest.approx(round((l - X0) / RES))
    assert ((Y0 - t) / RES) == pytest.approx(round((Y0 - t) / RES))
    assert d.window_transform.a == RES and not d.truncated_sides
    assert d.buffer_info["method"] == "dem_crs"


def test_buffer_truncated_by_dem_edge_is_reported(tmp_path):
    dem = write_dem(tmp_path / "d.tif", np.ones((100, 100), "float32"))
    g = write_gpkg(tmp_path / "p.gpkg", [pilot_box(2, 40, 50, 60)])
    cfg = cfg_for(tmp_path, dem, g, **{"processing.buffer_distance": 900})
    w = WarningCollector()
    d = prepare_spatial_domain(read_dem_info(dem), cfg, w)
    assert d.truncated_sides == ["west"]
    assert any("truncated" in m for m in w.messages)


def test_zero_buffer_warns_and_cells_units(tmp_path):
    dem = write_dem(tmp_path / "d.tif", np.ones((100, 100), "float32"))
    g = write_gpkg(tmp_path / "p.gpkg", [pilot_box(40, 40, 60, 60)])
    info = read_dem_info(dem)
    w = WarningCollector()
    prepare_spatial_domain(info, cfg_for(tmp_path, dem, g, **{"processing.buffer_distance": 0}), w)
    assert any("buffer_distance is 0" in m for m in w.messages)
    d = prepare_spatial_domain(
        info, cfg_for(tmp_path, dem, g, **{"processing.buffer_distance": 10, "processing.buffer_units": "cells"}),
        WarningCollector(),
    )
    assert d.window.width >= 20 + 2 * 10


def test_geographic_dem_buffer_is_computed_in_metres(tmp_path):
    res = 0.0009  # ~100 m
    arr = np.ones((200, 200), "float32")
    dem = write_dem(tmp_path / "geo.tif", arr, crs="EPSG:4326", transform=from_origin(-60.0, -3.0, res, res))
    pilot = box(-59.95, -3.15, -59.85, -3.05)
    g = write_gpkg(tmp_path / "gp.gpkg", [pilot], crs="EPSG:4326")
    cfg = cfg_for(tmp_path, dem, g, **{"processing.buffer_distance": 5000})
    d = prepare_spatial_domain(read_dem_info(dem), cfg, WarningCollector())
    assert d.buffer_info["method"] == "projected_crs" and "EPSG:327" in d.buffer_info["buffer_crs"]  # southern-hemisphere UTM
    grew_deg = pilot.bounds[0] - d.requested_bounds[0]
    assert grew_deg == pytest.approx(5000 / 111_320 / np.cos(np.radians(3.1)), rel=0.05)


# --------------------------- DEM extraction & nodata --------------------------- #
def test_extraction_preserves_values_nodata_grid(tmp_path):
    arr = np.arange(100 * 100, dtype="float32").reshape(100, 100)
    arr[10:20, 10:20] = NODATA
    dem = write_dem(tmp_path / "d.tif", arr)
    g = write_gpkg(tmp_path / "p.gpkg", [pilot_box(30, 30, 70, 70)])
    cfg = cfg_for(tmp_path, dem, g, **{"processing.buffer_distance": 300})
    info, w = read_dem_info(dem), WarningCollector()
    d = prepare_spatial_domain(info, cfg, w)
    out = tmp_path / "extent.tif"
    meta = extract_processing_dem(info, d, out, w)
    with rasterio.open(out) as o, rasterio.open(dem) as s:
        assert o.crs == s.crs and o.nodata == NODATA and o.res == s.res
        sub = s.read(1, window=d.window)
        assert np.array_equal(o.read(1), sub)
        assert o.transform == s.window_transform(d.window)
    assert meta["valid_cells"] == sub.size - int((sub == NODATA).sum())


def test_integer_dem_becomes_float32_and_nan_nodata_gets_sentinel(tmp_path):
    ints = np.arange(2500, dtype="int16").reshape(50, 50)
    dem = write_dem(tmp_path / "i.tif", ints, nodata=-32768)
    g = write_gpkg(tmp_path / "p.gpkg", [pilot_box(10, 10, 40, 40)])
    info, w = read_dem_info(dem), WarningCollector()
    d = prepare_spatial_domain(info, cfg_for(tmp_path, dem, g), w)
    meta = extract_processing_dem(info, d, tmp_path / "o.tif", w)
    assert meta["dtype"] == "float32"

    f = np.ones((50, 50), "float32")
    f[:5] = np.nan
    dem2 = write_dem(tmp_path / "n.tif", f, nodata=np.nan)
    info2 = read_dem_info(dem2)
    d2 = prepare_spatial_domain(info2, cfg_for(tmp_path, dem2, g), w)
    extract_processing_dem(info2, d2, tmp_path / "o2.tif", w)
    with rasterio.open(tmp_path / "o2.tif") as o:
        assert o.nodata == -9999.0
        assert (o.read(1)[:5] == -9999.0).all()  # NaN cells became the explicit sentinel
    assert any("NaN" in m for m in w.messages)


def test_all_nodata_inside_pilot_is_an_error(tmp_path):
    arr = np.ones((100, 100), "float32")
    arr[30:70, 30:70] = NODATA
    dem = write_dem(tmp_path / "d.tif", arr)
    g = write_gpkg(tmp_path / "p.gpkg", [pilot_box(40, 40, 60, 60)])
    info, w = read_dem_info(dem), WarningCollector()
    d = prepare_spatial_domain(info, cfg_for(tmp_path, dem, g), w)
    with pytest.raises(InputError, match="nodata"):
        extract_processing_dem(info, d, tmp_path / "o.tif", w)


# ------------------------------ cell geometry --------------------------------- #
def test_cell_geometry_projected_feet_and_geographic():
    ft = pyproj.CRS.from_epsg(2263)  # NAD83 / New York Long Island (ftUS)
    c = cell_geometry(ft, Affine(100, 0, 0, 0, -100, 0), (0, 0, 1000, 1000))
    assert c.dx_m == pytest.approx(30.48, rel=1e-3) and not c.geographic
    geo = pyproj.CRS.from_epsg(4326)
    g = cell_geometry(geo, from_origin(-60, 10, 0.001, 0.001), (-60, 9, -59, 10))
    assert g.geographic and g.dx_m == pytest.approx(111.3 * np.cos(np.radians(9.5)), rel=0.01)
    assert g.area_variation_fraction > 0
