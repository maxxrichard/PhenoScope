"""Instance-segmentation metrics.

``AP@t = TP / (TP + FP + FN)`` with one-to-one Hungarian matching at IoU >= t,
the definition used by Cellpose, StarDist and the Data Science Bowl 2018. We
report AP@0.5 as the headline number and mAP over IoU 0.5:0.95 for detail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU between every GT object (rows) and predicted object (cols), background excluded."""
    gt = gt.astype(np.int64).ravel()
    pred = pred.astype(np.int64).ravel()
    n_gt, n_pred = int(gt.max()) + 1, int(pred.max()) + 1
    overlap = np.bincount(gt * n_pred + pred, minlength=n_gt * n_pred).reshape(n_gt, n_pred)
    area_gt = overlap.sum(axis=1, keepdims=True)
    area_pred = overlap.sum(axis=0, keepdims=True)
    union = area_gt + area_pred - overlap
    iou = overlap / np.maximum(union, 1)
    return iou[1:, 1:]


def match_instances(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` under optimal one-to-one matching at ``threshold``."""
    iou = iou_matrix(gt, pred)
    n_gt, n_pred = iou.shape
    if n_gt == 0 or n_pred == 0:
        return 0, n_pred, n_gt
    # Restrict to non-empty labels (ids may be non-contiguous).
    gt_present = np.isin(np.arange(1, n_gt + 1), gt)
    pred_present = np.isin(np.arange(1, n_pred + 1), pred)
    iou = iou[gt_present][:, pred_present]
    n_gt, n_pred = iou.shape
    rows, cols = linear_sum_assignment(-iou)
    tp = int((iou[rows, cols] >= threshold).sum())
    return tp, n_pred - tp, n_gt - tp


def average_precision(
    gt: np.ndarray, pred: np.ndarray, thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> list[float]:
    out = []
    for t in thresholds:
        tp, fp, fn = match_instances(gt, pred, t)
        denom = tp + fp + fn
        out.append(tp / denom if denom else 1.0)
    return out


@dataclass
class InstanceMetrics:
    ap50: float
    ap75: float
    map: float
    precision50: float
    recall50: float
    f1_50: float
    n_gt: int
    n_pred: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_instances(gt: np.ndarray, pred: np.ndarray) -> InstanceMetrics:
    aps = average_precision(gt, pred, DEFAULT_THRESHOLDS)
    tp, fp, fn = match_instances(gt, pred, 0.5)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return InstanceMetrics(
        ap50=aps[0],
        ap75=aps[DEFAULT_THRESHOLDS.index(0.75)],
        map=float(np.mean(aps)),
        precision50=precision,
        recall50=recall,
        f1_50=f1,
        n_gt=int(len(np.unique(gt[gt > 0]))),
        n_pred=int(len(np.unique(pred[pred > 0]))),
    )


def aggregate_metrics(metrics: Sequence[InstanceMetrics]) -> dict[str, float]:
    keys = ["ap50", "ap75", "map", "precision50", "recall50", "f1_50"]
    return {k: float(np.mean([getattr(m, k) for m in metrics])) for k in keys}
