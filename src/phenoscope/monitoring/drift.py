"""Data/model drift monitoring for imaging batches.

Two complementary signals are checked for each new batch against a frozen
reference (the benchmark set a model was validated on):

* **Feature drift** – PSI and two-sample KS on interpretable per-image features
  (channel intensity percentiles, background, cell density, object area). A
  staining-batch change typically shows up here first.
* **Embedding drift** – kernel MMD with a permutation test on ViT embeddings,
  which catches shifts in texture/morphology not captured by summary stats.

A batch is flagged when either signal exceeds its threshold; the report lists the
top drifting features so analysts can tell a staining shift from a biological one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN, PSI_ALERT = 0.1, 0.25


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int | None = None) -> float:
    """PSI over reference-quantile bins.

    PSI is biased upward for small samples (roughly ``(bins-1) * (1/n + 1/m)`` under
    no drift), so by default the bin count shrinks to keep ~10 samples per bin.
    """
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if bins is None:
        bins = int(np.clip(min(len(ref), len(cur)) // 10, 3, 10))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_drift(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    res = ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def _rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def k(a, b):
        d = (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T
        return np.exp(-gamma * np.maximum(d, 0))

    n, m = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


def mmd_permutation_test(
    x: np.ndarray, y: np.ndarray, n_perm: int = 200, max_n: int = 500, seed: int = 0
) -> tuple[float, float]:
    """Unbiased RBF-MMD² with median-heuristic bandwidth; returns ``(mmd2, p_value)``."""
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(max_n, len(x)), replace=False)]
    y = y[rng.choice(len(y), min(max_n, len(y)), replace=False)]
    z = np.vstack([x, y]).astype(np.float64)
    sub = z[rng.choice(len(z), min(len(z), 300), replace=False)]
    d = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
    med = np.median(d[d > 0]) if (d > 0).any() else 1.0
    gamma = 1.0 / (2 * med**2)

    stat = _rbf_mmd2(x, y, gamma)
    n = len(x)
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(z))
        null.append(_rbf_mmd2(z[idx[:n]], z[idx[n:]], gamma))
    p = (1 + sum(s >= stat for s in null)) / (1 + n_perm)
    return stat, float(p)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float
    ks_p: float

    @property
    def level(self) -> str:
        if self.psi >= PSI_ALERT:
            return "alert"
        return "warn" if self.psi >= PSI_WARN else "ok"


@dataclass
class DriftReport:
    batch_id: str
    drifted: bool
    features: list[FeatureDrift]
    mmd2: float | None = None
    mmd_p: float | None = None
    reasons: list[str] = field(default_factory=list)

    def top(self, k: int = 5) -> list[FeatureDrift]:
        return sorted(self.features, key=lambda f: f.psi, reverse=True)[:k]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["features"] = [{**asdict(f), "level": f.level} for f in self.features]
        return d


class DriftMonitor:
    """Holds a reference distribution; :meth:`check` compares a new batch against it."""

    def __init__(
        self,
        reference_features: pd.DataFrame,
        reference_embeddings: np.ndarray | None = None,
        features: Sequence[str] | None = None,
        psi_threshold: float = PSI_ALERT,
        ks_alpha: float = 0.01,
        min_alert_features: int = 2,
        mmd_alpha: float = 0.01,
    ) -> None:
        self.ref = reference_features
        self.ref_emb = reference_embeddings
        self.features = list(features or reference_features.select_dtypes("number").columns)
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.min_alert_features = min_alert_features
        self.mmd_alpha = mmd_alpha

    def check(
        self,
        batch_features: pd.DataFrame,
        batch_embeddings: np.ndarray | None = None,
        batch_id: str = "batch",
    ) -> DriftReport:
        results = []
        for f in self.features:
            if f not in batch_features:
                continue
            r, c = self.ref[f].to_numpy(float), batch_features[f].to_numpy(float)
            ks, p = ks_drift(r, c)
            results.append(FeatureDrift(f, population_stability_index(r, c), ks, p))

        reasons = []
        # A feature alerts only when the shift is both large (PSI) and significant (KS),
        # which keeps small batches from tripping on sampling noise.
        alerts = [r for r in results if r.psi >= self.psi_threshold and r.ks_p < self.ks_alpha]
        if len(alerts) >= self.min_alert_features:
            reasons.append(
                f"{len(alerts)} features with PSI>={self.psi_threshold}: "
                + ", ".join(a.feature for a in sorted(alerts, key=lambda a: -a.psi)[:5])
            )

        mmd2 = mmd_p = None
        if self.ref_emb is not None and batch_embeddings is not None:
            mmd2, mmd_p = mmd_permutation_test(self.ref_emb, batch_embeddings)
            if mmd_p < self.mmd_alpha:
                reasons.append(f"embedding MMD²={mmd2:.4f} (p={mmd_p:.3g})")

        return DriftReport(batch_id, bool(reasons), results, mmd2, mmd_p, reasons)


def image_qc_features(image: np.ndarray, channel_names: Sequence[str]) -> dict[str, float]:
    """Cheap per-image statistics used as drift features (no segmentation needed)."""
    out: dict[str, float] = {}
    for c, name in enumerate(channel_names):
        plane = image[c].astype(np.float32)
        p1, p50, p99 = np.percentile(plane, [1, 50, 99])
        out[f"{name}_p01"] = float(p1)
        out[f"{name}_p50"] = float(p50)
        out[f"{name}_p99"] = float(p99)
        out[f"{name}_dynamic_range"] = float(p99 - p1)
        out[f"{name}_saturated_frac"] = float((plane >= plane.max()).mean())
    return out
