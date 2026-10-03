from pathlib import Path

from floodplain_delineation import hydrology


def test_hydrology_invokes_expected_whitebox_operations(tmp_path, monkeypatch):
    calls = []

    class FakeRunner:
        def __init__(self, work_dir):
            pass

        def run(self, tool, args):
            calls.append((tool, args))

    monkeypatch.setattr(hydrology, "WhiteboxRunner", FakeRunner)
    hydrology.run_hydrology(
        tmp_path / "dem.tif",
        tmp_path / "conditioned.tif",
        tmp_path / "pointer.tif",
        tmp_path / "accum.tif",
        tmp_path / "streams.tif",
        1000,
        tmp_path / "streams.shp",
    )
    assert [name for name, _ in calls] == [
        "fill_depressions",
        "d8_pointer",
        "d8_flow_accumulation",
        "extract_streams",
        "raster_streams_to_vector",
    ]
    assert "--pntr" in calls[2][1]
    assert "--threshold=1000" in calls[3][1]
