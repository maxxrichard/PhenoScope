"""Fusing image embeddings with transcriptomics (and clinical covariates).

Image profiles (sample-level aggregated ViT embeddings) and gene-expression
profiles are each reduced with PCA, then combined by:

* ``early``  – concatenation of the reduced blocks into one classifier;
* ``late``   – average of per-modality classifier probabilities (stacking-free).

:func:`compare_fusion` reports cross-validated AUROC for image-only,
expression-only and fused models with grouped folds (e.g. by donor or plate),
so leakage across technical replicates does not inflate the gain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def align_modalities(
    image: pd.DataFrame, expression: pd.DataFrame, key: str = "sample_id"
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Inner-join two sample-indexed tables; returns aligned matrices and the shared ids."""
    img = image.set_index(key) if key in image.columns else image
    expr = expression.set_index(key) if key in expression.columns else expression
    shared = img.index.intersection(expr.index)
    if len(shared) == 0:
        raise ValueError("no overlapping samples between modalities")
    return (
        img.loc[shared].to_numpy(np.float32),
        np.log1p(expr.loc[shared].clip(lower=0)).to_numpy(np.float32),
        shared,
    )


def _block(n_components: int):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=0))


def _clf(c: float = 0.5):
    return LogisticRegression(C=c, max_iter=2000, class_weight="balanced")


@dataclass
class FusionReport:
    auroc_image: float
    auroc_expression: float
    auroc_early: float
    auroc_late: float
    n_samples: int

    @property
    def gain_over_image(self) -> float:
        return max(self.auroc_early, self.auroc_late) - self.auroc_image

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "gain_over_image": self.gain_over_image}


def compare_fusion(
    x_img: np.ndarray,
    x_expr: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_components: int = 32,
    n_splits: int = 5,
    seed: int = 0,
) -> FusionReport:
    """Out-of-fold AUROC for image-only, expression-only, early and late fusion (binary y)."""
    y = np.asarray(y)
    if groups is not None:
        splits = GroupKFold(n_splits=n_splits).split(x_img, y, groups)
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x_img, y)

    oof = {k: np.zeros(len(y)) for k in ("image", "expr", "early", "late")}
    for tr, te in splits:
        k_img = min(n_components, x_img.shape[1], len(tr) - 1)
        k_expr = min(n_components, x_expr.shape[1], len(tr) - 1)
        b_img, b_expr = _block(k_img).fit(x_img[tr]), _block(k_expr).fit(x_expr[tr])
        zi_tr, zi_te = b_img.transform(x_img[tr]), b_img.transform(x_img[te])
        ze_tr, ze_te = b_expr.transform(x_expr[tr]), b_expr.transform(x_expr[te])

        m_img = _clf().fit(zi_tr, y[tr])
        m_expr = _clf().fit(ze_tr, y[tr])
        m_early = _clf().fit(np.hstack([zi_tr, ze_tr]), y[tr])
        p_img = m_img.predict_proba(zi_te)[:, 1]
        p_expr = m_expr.predict_proba(ze_te)[:, 1]
        oof["image"][te] = p_img
        oof["expr"][te] = p_expr
        oof["early"][te] = m_early.predict_proba(np.hstack([zi_te, ze_te]))[:, 1]
        oof["late"][te] = 0.5 * (p_img + p_expr)

    return FusionReport(
        auroc_image=float(roc_auc_score(y, oof["image"])),
        auroc_expression=float(roc_auc_score(y, oof["expr"])),
        auroc_early=float(roc_auc_score(y, oof["early"])),
        auroc_late=float(roc_auc_score(y, oof["late"])),
        n_samples=len(y),
    )
