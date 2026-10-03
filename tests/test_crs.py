import pytest
from pyproj import CRS

from floodplain_delineation.crs import validate_input_crs, validate_processing_crs
from floodplain_delineation.exceptions import ConfigurationError, InputError


def test_matching_input_crs_is_accepted():
    crs = CRS.from_epsg(4326)
    assert validate_input_crs(crs, crs, "DEM") == crs


def test_mismatched_input_crs_is_rejected():
    with pytest.raises(InputError, match="does not match"):
        validate_input_crs(CRS.from_epsg(3857), CRS.from_epsg(4326), "DEM")


def test_projected_metric_crs_is_accepted():
    validate_processing_crs(CRS.from_epsg(31983))


def test_geographic_crs_is_rejected():
    with pytest.raises(ConfigurationError, match="projected"):
        validate_processing_crs(CRS.from_epsg(4326))
