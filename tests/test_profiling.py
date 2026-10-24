import numpy as np
import pandas as pd

from phenoscope.profiling import (
    aggregate_profiles,
    extract_features,
    extract_object_crops,
    feature_select,
    normalize_to_controls,
)


def test_extract_features_columns(cells):
    img, gt = cells
    df = extract_features(img, gt, ["DAPI", "CellMask"], pixel_size_um=0.5, texture=True)
    assert len(df) == gt.max()
    for col in ("area", "area_um2", "form_factor", "DAPI_intensity_mean", "CellMask_intensity_std",
                "DAPI_intensity_integrated", "DAPI_texture_contrast"):
        assert col in df.columns
    assert np.allclose(df["area_um2"], df["area"] * 0.25)


def test_object_crops(cells):
    img, gt = cells
    crops, ids = extract_object_crops(img, gt, size=32)
    assert crops.shape == (gt.max(), 2, 32, 32)
    assert sorted(ids) == list(range(1, gt.max() + 1))


def test_aggregate_and_normalize():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "plate": ["p1"] * 40,
        "well": np.repeat(["A01", "A02", "B01", "B02"], 10),
        "f1": rng.normal(0, 1, 40),
        "f2": rng.normal(5, 2, 40),
    })
    prof = aggregate_profiles(df, features=["f1", "f2"])
    assert list(prof["n_cells"]) == [10, 10, 10, 10]
    ctrl = prof["well"].isin(["A01", "A02"])
    norm = normalize_to_controls(prof, ["f1", "f2"], ctrl)
    assert norm.loc[ctrl, "f1"].abs().max() < 1.5


def test_feature_select_drops_correlated():
    x = np.arange(50, dtype=float)
    df = pd.DataFrame({"a": x, "b": 2 * x + 1, "c": np.sin(x), "d": np.ones(50)})
    assert feature_select(df, ["a", "b", "c", "d"]) == ["a", "c"]
