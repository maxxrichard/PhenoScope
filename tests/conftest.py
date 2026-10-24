from __future__ import annotations

import pytest

from phenoscope.benchmarks import synthetic_cells


@pytest.fixture
def cells():
    return synthetic_cells((256, 256), n_cells=40, seed=1)


@pytest.fixture
def zarr_image(tmp_path, cells):
    from phenoscope.io import write_labels, write_ome_zarr

    img, gt = cells
    uri = str(tmp_path / "plate1_A01.zarr")
    write_ome_zarr(img, uri, channel_names=["DAPI", "CellMask"], pixel_size_um=(0.65, 0.65))
    write_labels(gt, uri, name="ground_truth")
    return uri, img, gt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PHENOSCOPE_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    from phenoscope.config import get_settings
    from phenoscope.db.session import get_engine

    get_settings.cache_clear()
    get_engine.cache_clear()
    from fastapi.testclient import TestClient

    from phenoscope.api.main import create_app

    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()
    get_engine.cache_clear()
