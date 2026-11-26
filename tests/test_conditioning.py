"""breach_least_cost conditioning (configurable limits, optional fill) and its integration."""

import numpy as np
import pytest
import rasterio

from conftest import NODATA, RES, X0, Y0, box, config_dict, requires_wbt, valley_terrain, write_dem, write_gpkg
from floodplain_delineation.config import parse_config
from floodplain_delineation.exceptions import ConfigError
from floodplain_delineation.pipeline import run_pipeline
from floodplain_delineation.preprocessing import conditioning_plan


def proc(tmp_path, conditioning):
    return parse_config(config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg", **{"processing.conditioning": conditioning})).processing


# ------------------------------------------------------------------ configuration / flags ------------------
def test_flags_follow_the_configuration_exactly(tmp_path):
    full = proc(tmp_path, {"method": "breach_least_cost", "fill_remaining": True, "breach_least_cost": {
        "max_distance_cells": 200, "max_cost": 150.5, "minimize_distance": True, "flat_increment": 0.001}})
    steps = conditioning_plan(full)
    # breach (every configured limit) and then an explicit fill step; the tool's own --fill flag is not used
    assert steps == [("BreachDepressionsLeastCost", ["--dist=200", "--max_cost=150.5", "--min_dist", "--flat_increment=0.001"]),
                     ("FillDepressionsWangAndLiu", ["--fix_flats"])]
    assert all("--fill" not in flags for _, flags in steps)
    minimal = proc(tmp_path, {"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 50}})
    assert conditioning_plan(minimal) == [("BreachDepressionsLeastCost", ["--dist=50"])]           # nothing invented, no fill
    assert conditioning_plan(proc(tmp_path, {"method": "breach"})) == [("BreachDepressions", [])]
    assert conditioning_plan(proc(tmp_path, {"method": "fill"}))[0][0] == "FillDepressionsWangAndLiu"


@pytest.mark.parametrize("cond, match", [
    ({"method": "breach_least_cost"}, "required"),
    ({"method": "breach_least_cost", "breach_least_cost": {}}, "max_distance_cells"),
    ({"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 0}}, "max_distance_cells"),
    ({"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 10.5}}, "whole number"),
    ({"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": True}}, "max_distance_cells"),
    ({"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 10, "max_cost": -1}}, "max_cost"),
    ({"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 10, "minimize_distance": "yes"}}, "minimize_distance"),
    ({"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 10, "bogus": 1}}, "bogus"),
    ({"method": "breach_least_cost", "fill_remaining": "x", "breach_least_cost": {"max_distance_cells": 10}}, "fill_remaining"),
    ({"method": "breach", "breach_least_cost": {"max_distance_cells": 10}}, "only apply"),      # no silent ignoring
    ({"method": "fill", "fill_remaining": True}, "only apply"),
])
def test_invalid_least_cost_configuration_is_rejected(tmp_path, cond, match):
    with pytest.raises(ConfigError, match=match):
        proc(tmp_path, cond)


def test_effective_configuration_echoes_the_least_cost_parameters(tmp_path):
    cfg = parse_config(config_dict(tmp_path, tmp_path / "d.tif", tmp_path / "p.gpkg", **{"processing.conditioning": {
        "method": "breach_least_cost", "fill_remaining": True,
        "breach_least_cost": {"max_distance_cells": 80, "minimize_distance": True}}}))
    c = cfg.to_dict()["processing"]["conditioning"]
    assert c["breach_least_cost"] == {"max_distance_cells": 80, "max_cost": None, "minimize_distance": True, "flat_increment": None}
    assert c["fill_remaining"] is True


# ------------------------------------------------------------------ real WhiteboxTools -----------------------
pytestmark_wbt = [requires_wbt, pytest.mark.wbt]


def pit_dataset(tmp_path):
    """Valley plus a closed 6x6 pit (10 m deep, high rim 3 cells wide) that a 2-cell breach cannot cross."""
    arr = valley_terrain()
    arr[10:19, 70:79] = arr[10:19, 70:79] + 12.0
    arr[13:16, 73:76] = arr[13:16, 73:76] - 25.0
    dem = write_dem(tmp_path / "dem.tif", arr)
    h, w = arr.shape
    gpkg = write_gpkg(tmp_path / "p.gpkg", [box(X0 + 5 * RES, Y0 - (h - 5) * RES, X0 + (w - 5) * RES, Y0 - 5 * RES)])
    return dem, gpkg


@requires_wbt
@pytest.mark.wbt
def test_least_cost_breach_runs_and_passes_the_configured_flags(tmp_path):
    dem, gpkg = pit_dataset(tmp_path)
    cond = {"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 15, "max_cost": 500.0, "minimize_distance": True}}
    s = run_pipeline(parse_config(config_dict(tmp_path, dem, gpkg, **{"processing.conditioning": cond})))
    call = s["whitebox_calls"][0]
    assert call["tool"] == "BreachDepressionsLeastCost"
    assert call["flags"][:1] == [f"--dem={tmp_path / 'out' / 'intermediate' / 'dem_processing_crs.tif'}"]
    assert call["flags"][1:4] == ["--dist=15", "--max_cost=500.0", "--min_dist"]          # exactly what was configured
    assert call["flags"][4].startswith("--output=") and call["flags"][4].endswith("conditioned_dem.wbt_raw.tif") and len(call["flags"]) == 5
    assert s["stages"]["conditioning"]["tool"] == "BreachDepressionsLeastCost" and s["status"] == "success"
    assert s["configuration"]["effective"]["processing"]["conditioning"]["breach_least_cost"]["max_distance_cells"] == 15


@requires_wbt
@pytest.mark.wbt
def test_breach_then_fill_fills_what_the_bounded_breach_could_not_cross(tmp_path):
    dem, gpkg = pit_dataset(tmp_path)
    base = {"method": "breach_least_cost", "breach_least_cost": {"max_distance_cells": 2}}
    no_fill = run_pipeline(parse_config(config_dict(tmp_path / "a", dem, gpkg, **{"processing.conditioning": base})))
    fill = run_pipeline(parse_config(config_dict(tmp_path / "b", dem, gpkg, **{"processing.conditioning": {**base, "fill_remaining": True}})))
    assert [c["tool"] for c in no_fill["whitebox_calls"][:1]] == ["BreachDepressionsLeastCost"]
    assert [c["tool"] for c in fill["whitebox_calls"][:2]] == ["BreachDepressionsLeastCost", "FillDepressionsWangAndLiu"]
    a, b = no_fill["stages"]["conditioning"], fill["stages"]["conditioning"]
    # breaching lowers terrain; the tool also raises a few cells by a tiny flat-area increment (< 0.1 here), nothing more
    assert 0.0 <= a["max_elevation_increase"] < 0.1 and a["max_elevation_decrease"] > 0
    assert b["max_elevation_increase"] > 10.0                       # the unbreached pit (12 m rim, 25 m hole) was filled
    assert b["max_elevation_increase"] <= 40.0 and b["tool"] == "BreachDepressionsLeastCost + FillDepressionsWangAndLiu"
    s1, s2 = b["steps"]                                             # each step is reported separately
    assert s1["max_elevation_increase"] < 0.1 and s1["max_elevation_decrease"] > 0
    assert s2["max_elevation_increase"] > 10.0 and s2["max_elevation_decrease"] == 0.0     # filling never lowers terrain
    # the filled pit has a defined outlet: fewer cells without a downstream neighbour than with the pit left in place
    assert fill["stages"]["flow_routing"]["direction"]["cells_without_direction"] < no_fill["stages"]["flow_routing"]["direction"]["cells_without_direction"]
    assert not list((tmp_path / "b" / "out" / "intermediate").rglob("conditioning_step*"))   # temporary raster removed
