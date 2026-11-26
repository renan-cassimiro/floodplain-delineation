"""Global processing CRS / resolution / resampling (Stage 0): configuration, grid construction, DEM warp."""

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject

from conftest import NODATA, config_dict, write_dem
from floodplain_delineation import rasters, reprojection
from floodplain_delineation.config import DEFAULT_PROCESSING_CRS, parse_config
from floodplain_delineation.exceptions import ConfigError, InputError
from floodplain_delineation.inputs import prepare_spatial_domain, read_dem_info
from floodplain_delineation.logging_utils import WarningCollector
from floodplain_delineation.reprojection import reproject_dem
from floodplain_delineation.spatial import resolve_processing_crs

ALBERS = RioCRS.from_string("ESRI:102033")


# ------------------------------------------------------------------ CRS ------------------
def test_default_processing_crs_is_esri_102033_and_is_documented_correctly(tmp_path):
    cfg = parse_config(config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg"))
    assert DEFAULT_PROCESSING_CRS == "ESRI:102033"
    c = resolve_processing_crs(DEFAULT_PROCESSING_CRS)
    d = c.describe()
    assert d["authority_code"] == "ESRI:102033" and "EPSG" not in d["authority_code"]     # never relabelled as EPSG
    assert d["horizontal_units"] == "metre" and d["metres_per_unit"] == 1.0 and c.pcrs.is_projected
    assert "Albers" in d["name"] and d["type"] == "Projected CRS"
    default = config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg"); del default["processing"]["crs"]
    assert parse_config(default).processing.crs == "ESRI:102033"


@pytest.mark.parametrize("spec, match", [
    ("EPSG:4326", "projected"),                      # geographic: rejected for Euclidean IDW distances
    ("EPSG:102033", "not recognised"),               # a different (non-existent) identifier: no silent substitution
    ("not-a-crs", "not recognised"),
])
def test_invalid_processing_crs_is_rejected_with_a_clear_message(tmp_path, spec, match):
    with pytest.raises(ConfigError, match=match):
        parse_config(config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg", **{"processing.crs": spec}))


def test_other_projected_crs_and_units_are_accepted(tmp_path):
    ft = resolve_processing_crs("EPSG:2263")         # NY Long Island, US survey feet
    assert ft.unit_factor == pytest.approx(0.3048006, rel=1e-5) and "foot" in ft.unit_name.lower()


@pytest.mark.parametrize("key, value", [("processing.target_resolution", 0), ("processing.target_resolution", -3.0),
                                        ("processing.target_resolution", "30"), ("processing.dem_resampling", "mode")])
def test_resolution_and_resampling_are_validated(tmp_path, key, value):
    with pytest.raises(ConfigError):
        parse_config(config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg", **{key: value}))


@pytest.mark.parametrize("rem, ok", [
    ({"enabled": True, "sampling_interval_pixels": 10, "idw_power": 2.0}, True),
    ({"enabled": False}, True),
    ({"enabled": True}, False),                                              # interval is required
    ({"enabled": True, "sampling_interval_pixels": 0}, False),
    ({"enabled": True, "sampling_interval_pixels": 2.5}, False),
    ({"enabled": True, "sampling_interval_pixels": True}, False),
    ({"enabled": True, "sampling_interval_pixels": 5, "idw_power": 0}, False),
    ({"enabled": True, "sampling_interval_pixels": 5, "idw_power": -1}, False),
    ({"enabled": True, "sampling_interval_pixels": 5, "idw_power": float("inf")}, False),
    ({"enabled": True, "sampling_interval_pixels": 5, "distance_crs": "EPSG:5880"}, False),   # parameter removed by design
])
def test_rem_configuration_validation(tmp_path, rem, ok):
    cfgd = config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg", **{"processing.rem": rem})
    if ok:
        r = parse_config(cfgd).processing.rem
        assert r.idw_power == rem.get("idw_power", 2.0) and r.enabled is rem["enabled"]
    else:
        with pytest.raises(ConfigError):
            parse_config(cfgd)


def test_rem_is_disabled_by_default_so_existing_yaml_files_keep_working(tmp_path):
    cfgd = config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg")
    assert "rem" not in cfgd["processing"] and parse_config(cfgd).processing.rem.enabled is False


# --------------------------------------------------------------- warp ------------------
def geographic_dem(tmp_path, hole=True):
    """0.001-degree DEM (~110 m) over the Xingu region with a smooth surface and a nodata block."""
    h, w = 160, 220
    y, x = np.mgrid[0:h, 0:w].astype("float64")
    arr = (200 + 0.8 * x + 0.5 * y + 6 * np.sin(x / 15) + 4 * np.cos(y / 11)).astype("float32")
    if hole:
        arr[60:90, 100:140] = NODATA
    path = write_dem(tmp_path / "geo.tif", arr, crs="EPSG:4326", transform=from_origin(-52.5, -8.0, 0.001, 0.001))
    return path, arr


def domain_for(tmp_path, dem, **over):
    cfg = parse_config(config_dict(tmp_path, dem, tmp_path / "none.gpkg",
                                   **{"inputs.pilot_area": None, "processing.crs": "ESRI:102033", **over}))
    info, w = read_dem_info(dem), WarningCollector()
    return info, prepare_spatial_domain(info, cfg, w), w


def test_grid_covers_the_densified_footprint_is_snapped_and_resolution_is_documented(tmp_path):
    dem, _ = geographic_dem(tmp_path)
    info, d, w = domain_for(tmp_path, dem)
    g = d.grid
    suggested, _, _ = calculate_default_transform("EPSG:4326", ALBERS, info.width, info.height, *info.bounds)
    assert g.res_x == g.res_y == pytest.approx(abs(suggested.a)) and g.crs == ALBERS
    assert not g.identical_to_source and d.resolution_info["source"].startswith("derived")
    assert (g.transform.c / g.res_x) == pytest.approx(round(g.transform.c / g.res_x), abs=1e-6)     # lattice at the CRS origin
    assert (g.transform.f / g.res_y) == pytest.approx(round(g.transform.f / g.res_y), abs=1e-6)
    # every point of the (densified) source boundary falls inside the grid
    tr = pyproj.Transformer.from_crs("EPSG:4326", "ESRI:102033", always_xy=True)
    l, b, r, t = info.bounds
    t01 = np.linspace(0, 1, 41)
    edge = np.concatenate([np.column_stack([l + (r - l) * t01, np.full(41, b)]), np.column_stack([np.full(41, r), b + (t - b) * t01]),
                           np.column_stack([l + (r - l) * t01, np.full(41, t)]), np.column_stack([np.full(41, l), b + (t - b) * t01])])
    ex, ey = tr.transform(edge[:, 0], edge[:, 1])
    gl, gb, gr, gt = g.bounds
    assert min(ex) >= gl - g.res_x and max(ex) <= gr + g.res_x and min(ey) >= gb - g.res_y and max(ey) <= gt + g.res_y
    assert any("no pilot" in m.lower() for m in w.messages)


def test_configured_target_resolution_is_used_exactly(tmp_path):
    dem, _ = geographic_dem(tmp_path)
    _, d, w = domain_for(tmp_path, dem, **{"processing.target_resolution": 90.0})
    assert d.grid.res_x == d.grid.res_y == 90.0 and "configured" in d.resolution_info["source"]
    assert d.resolution_info["target_to_source_cell_size_ratio"] == pytest.approx(90 / 110, rel=0.05)
    _, d2, w2 = domain_for(tmp_path, dem, **{"processing.target_resolution": 400.0})
    assert any("coarser than the source DEM" in m for m in w2.messages)


@pytest.mark.parametrize("resampling", ["bilinear", "near", "cubic"])
def test_reprojected_dem_equals_the_one_shot_warp_and_never_invents_data(tmp_path, monkeypatch, resampling):
    dem, arr = geographic_dem(tmp_path)
    info, d, w = domain_for(tmp_path, dem, **{"processing.dem_resampling": resampling, "processing.target_resolution": 100.0})
    monkeypatch.setattr(reprojection, "WARP_STRIPE_ROWS", 10 ** 6)          # whole grid in one warp window
    out = tmp_path / "dem_processing_crs.tif"
    res = reproject_dem(info, d, out, resampling, w)
    g = d.grid
    with rasterio.open(out) as o:
        got = o.read(1)
        assert o.crs == ALBERS and o.transform == g.transform and o.shape == (g.height, g.width) and o.nodata == NODATA
    method = {"bilinear": Resampling.bilinear, "near": Resampling.nearest, "cubic": Resampling.cubic}[resampling]
    ref = np.full(got.shape, NODATA, "float32"); near = ref.copy()
    for dst, m in ((ref, method), (near, Resampling.nearest)):
        reproject(arr, dst, src_transform=info.transform, src_crs=info.crs, dst_transform=g.transform, dst_crs=g.crs,
                  src_nodata=NODATA, dst_nodata=NODATA, resampling=m)
    valid, vref, vnear = got != NODATA, ref != NODATA, near != NODATA
    assert not (valid & ~vnear).any()                                        # no data where the nearest source cell is nodata
    both = valid & vref
    assert both.sum() > 0.9 * valid.sum()
    assert np.abs(got[both] - ref[both]).max() < 1e-3                        # one window == the one-shot GDAL warp
    # the nodata block (source rows 60-89, cols 100-139) stays nodata in its interior: nothing is interpolated across it
    cy, cx = rasterio.transform.rowcol(g.transform, *pyproj.Transformer.from_crs("EPSG:4326", "ESRI:102033", always_xy=True)
                                       .transform(-52.5 + 0.12 * 1.0, -8.0 - 0.075))
    assert got[cy, cx] == NODATA
    assert res["valid_cells"] == int(valid.sum()) and res["statistics_before_over_source_window"]["valid_cells"] > 0
    assert res["elevation_changed_by_resampling"] is True and res["warp_settings"]["stripe_rows_used"] >= 64
    assert any("reprojected/resampled" in m for m in w.messages)


@pytest.mark.parametrize("resampling", ["bilinear", "near"])
def test_warp_result_is_independent_of_the_memory_budget_and_of_stripe_boundaries_in_validity(tmp_path, monkeypatch, resampling):
    dem, arr = geographic_dem(tmp_path)
    info, d, w = domain_for(tmp_path, dem, **{"processing.dem_resampling": resampling, "processing.target_resolution": 100.0})
    monkeypatch.setattr(reprojection, "WARP_STRIPE_ROWS", 70)               # several warp stripes
    outs = []
    for budget in (2 * 2 ** 20, 400):                                       # very different memory budgets
        monkeypatch.setattr(rasters, "BLOCK_BYTES", budget)
        reproject_dem(info, d, tmp_path / f"o{budget}.tif", resampling, w)
        outs.append(rasterio.open(tmp_path / f"o{budget}.tif").read(1))
    assert np.array_equal(outs[0], outs[1])                                 # reproducible whatever the memory budget
    assert (outs[0] != NODATA).sum() > 0
    # decomposition only changes values a little (GDAL kernel scale) and never creates or removes data cells
    monkeypatch.setattr(reprojection, "WARP_STRIPE_ROWS", 10 ** 6)
    reproject_dem(info, d, tmp_path / "one.tif", resampling, w)
    one = rasterio.open(tmp_path / "one.tif").read(1)
    mism = ((outs[0] != NODATA) != (one != NODATA)).sum()
    assert mism <= 0.002 * (one != NODATA).sum()


def test_warp_settings_are_part_of_the_stage_fingerprint(tmp_path):
    from floodplain_delineation.reprojection import warp_settings, warp_stripe_rows
    assert warp_settings()["stripe_rows"] == reprojection.WARP_STRIPE_ROWS
    assert warp_stripe_rows(48316) == 1024 and warp_stripe_rows(10 ** 7) == 64 and warp_stripe_rows(100) == 1024


def test_source_file_is_never_modified_and_identity_case_is_a_pure_copy(tmp_path):
    dem, arr = geographic_dem(tmp_path)
    before = dem.read_bytes()
    info, d, w = domain_for(tmp_path, dem)
    reproject_dem(info, d, tmp_path / "o.tif", "bilinear", w)
    assert dem.read_bytes() == before
    # DEM already in the processing CRS and no target resolution: grid is a window of the DEM grid, values copied
    utm = write_dem(tmp_path / "utm.tif", np.arange(400, dtype="float32").reshape(20, 20))
    cfg = parse_config(config_dict(tmp_path, utm, tmp_path / "x.gpkg", **{"inputs.pilot_area": None}))
    info2, w2 = read_dem_info(utm), WarningCollector()
    d2 = prepare_spatial_domain(info2, cfg, w2)
    assert d2.grid.identical_to_source and d2.resolution_info["source"].startswith("native")
    res = reproject_dem(info2, d2, tmp_path / "copy.tif", "bilinear", w2)
    with rasterio.open(utm) as s, rasterio.open(tmp_path / "copy.tif") as o:
        assert np.array_equal(s.read(1), o.read(1)) and res["resampling"].startswith("none")


def test_extent_outside_the_projection_domain_is_reported(tmp_path):
    # a DEM that does not overlap the pilot is an input error, not a silent empty grid
    dem, _ = geographic_dem(tmp_path)
    from shapely.geometry import box
    from conftest import write_gpkg
    far = write_gpkg(tmp_path / "far.gpkg", [box(10, 10, 10.1, 10.1)], crs="EPSG:4326")
    cfg = parse_config(config_dict(tmp_path, dem, far, **{"processing.crs": "ESRI:102033"}))
    with pytest.raises(InputError, match="does not intersect"):
        prepare_spatial_domain(read_dem_info(dem), cfg, WarningCollector())
