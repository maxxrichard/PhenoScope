import numpy as np
import pandas as pd

from phenoscope.benchmarks import synthetic_cells
from phenoscope.fusion import align_modalities, compare_fusion
from phenoscope.monitoring import DriftMonitor, mmd_permutation_test, population_stability_index
from phenoscope.monitoring.drift import image_qc_features


def _qc(n, gain, seed0):
    rows = []
    for s in range(n):
        img, _ = synthetic_cells((128, 128), n_cells=15, stain_gain=gain, seed=seed0 + s)
        rows.append(image_qc_features(img, ["DAPI", "CellMask"]))
    return pd.DataFrame(rows)


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(size=5000), rng.normal(size=5000)) < 0.02
    assert population_stability_index(rng.normal(size=5000), rng.normal(1, 1, 5000)) > 0.25


def test_drift_monitor_flags_staining_shift():
    ref = _qc(30, 1.0, 0)
    same = _qc(30, 1.0, 100)
    shifted = _qc(30, 1.6, 200)  # brighter staining batch
    mon = DriftMonitor(ref)
    assert not mon.check(same, batch_id="same").drifted
    rep = mon.check(shifted, batch_id="stain-batch-2")
    assert rep.drifted
    assert any("p99" in f.feature or "p50" in f.feature for f in rep.top(3))


def test_mmd_detects_embedding_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 16)), rng.normal(size=(200, 16))
    c = rng.normal(0.5, 1, size=(200, 16))
    assert mmd_permutation_test(a, b, n_perm=100)[1] > 0.05
    assert mmd_permutation_test(a, c, n_perm=100)[1] < 0.05


def test_fusion_improves_auroc_when_modalities_complement():
    rng = np.random.default_rng(0)
    n = 300
    z_img, z_expr = rng.normal(size=n), rng.normal(size=n)
    y = (z_img + z_expr + rng.normal(0, 0.5, n) > 0).astype(int)
    x_img = np.c_[z_img, rng.normal(size=(n, 40))] + rng.normal(0, 0.1, (n, 41))
    x_expr = np.exp(np.c_[z_expr, rng.normal(size=(n, 60))] * 0.5 + 2)
    img_df = pd.DataFrame(x_img).assign(sample_id=[f"s{i}" for i in range(n)])
    expr_df = pd.DataFrame(x_expr).assign(sample_id=[f"s{i}" for i in range(n)])
    xi, xe, ids = align_modalities(img_df, expr_df)
    assert len(ids) == n
    rep = compare_fusion(xi, xe, y, n_components=16)
    assert rep.auroc_early > rep.auroc_image
    assert rep.gain_over_image > 0.03
