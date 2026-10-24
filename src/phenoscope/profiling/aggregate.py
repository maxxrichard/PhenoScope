"""Profile aggregation and normalisation (well/sample-level), pycytominer-style."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from phenoscope.preprocessing.normalize import robust_zscore


def aggregate_profiles(
    df: pd.DataFrame,
    by: Sequence[str] = ("plate", "well"),
    features: Sequence[str] | None = None,
    method: str = "median",
) -> pd.DataFrame:
    by = list(by)
    features = list(features or df.select_dtypes("number").columns.difference(by + ["label"]))
    agg = df.groupby(by)[features].agg(method)
    agg["n_cells"] = df.groupby(by).size()
    return agg.reset_index()


def normalize_to_controls(
    profiles: pd.DataFrame,
    features: Sequence[str],
    control_mask: pd.Series,
    by: str | None = "plate",
) -> pd.DataFrame:
    """Robust z-score each feature against negative-control wells (per plate if ``by``)."""
    out = profiles.copy()
    groups = [(None, out.index)] if by is None else out.groupby(by).groups.items()
    for _, idx in groups:
        block = out.loc[idx, list(features)].to_numpy(float)
        ctrl = out.loc[idx][control_mask.loc[idx]][list(features)].to_numpy(float)
        if len(ctrl) == 0:
            ctrl = block
        out.loc[idx, list(features)] = robust_zscore(block, ctrl)
    return out


def feature_select(
    profiles: pd.DataFrame,
    features: Sequence[str],
    corr_threshold: float = 0.9,
    min_variance: float = 1e-6,
) -> list[str]:
    """Drop near-constant and highly correlated features (greedy, keeps first seen)."""
    x = profiles[list(features)].astype(float)
    keep = [f for f in features if np.nanvar(x[f]) > min_variance]
    corr = x[keep].corr().abs().to_numpy()
    selected: list[int] = []
    for i in range(len(keep)):
        if all(corr[i, j] < corr_threshold for j in selected):
            selected.append(i)
    return [keep[i] for i in selected]
