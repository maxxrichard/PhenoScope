import numpy as np
from typer.testing import CliRunner

from phenoscope.cli import app
from phenoscope.io import read_ome_zarr

runner = CliRunner()


def test_convert_segment_features(tmp_path, cells):
    img, _ = cells
    np.save(tmp_path / "img.npy", img)
    uri = str(tmp_path / "img.zarr")
    r = runner.invoke(app, ["convert", str(tmp_path / "img.npy"), uri, "--channels", "DAPI,CellMask"])
    assert r.exit_code == 0, r.output
    assert read_ome_zarr(uri).channel_names == ["DAPI", "CellMask"]

    r = runner.invoke(app, ["segment", uri, "--channels", "DAPI"])
    assert r.exit_code == 0, r.output
    assert '"n_objects"' in r.output

    out = tmp_path / "feat.csv"
    r = runner.invoke(app, ["features", uri, str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()


def test_synthetic_benchmark():
    r = runner.invoke(app, ["benchmark", "synthetic"])
    assert r.exit_code == 0, r.output
    assert "watershed" in r.output
