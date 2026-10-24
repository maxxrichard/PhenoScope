"""Benchmark runner: evaluates one or more segmenters on a dataset and tabulates AP."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from phenoscope.evaluation.instance import evaluate_instances
from phenoscope.segmentation.base import get_segmenter

log = logging.getLogger(__name__)


def run_benchmark(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    segmenters: Mapping[str, Mapping[str, Any]],
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """Return a per-image metrics table and log the per-model mean.

    ``segmenters`` maps a display name to ``{"backend": <registry name>, **kwargs}``.
    """
    models = {
        name: get_segmenter(cfg.get("backend", name), **{k: v for k, v in cfg.items() if k != "backend"})
        for name, cfg in segmenters.items()
    }
    rows = []
    for i, (img, gt) in enumerate(pairs):
        for name, model in models.items():
            res = model.run(img)
            m = evaluate_instances(gt, res.labels)
            rows.append({"dataset": dataset_name, "image": i, "model": name,
                         "runtime_s": res.runtime_s, **m.as_dict()})
    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("model")[["ap50", "ap75", "map", "f1_50"]].mean()
        log.info("benchmark %s\n%s", dataset_name, summary.round(3).to_string())
    return df


def summarize(df: pd.DataFrame, baseline: str | None = None) -> pd.DataFrame:
    s = df.groupby("model")[["ap50", "ap75", "map", "f1_50", "runtime_s"]].mean()
    if baseline and baseline in s.index:
        s["ap50_delta_pp"] = (s["ap50"] - s.loc[baseline, "ap50"]) * 100
    return s.sort_values("ap50", ascending=False)
