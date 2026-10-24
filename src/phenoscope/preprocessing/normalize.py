"""Intensity normalisation helpers."""

from __future__ import annotations

import numpy as np


def percentile_normalize(
    img: np.ndarray,
    low: float = 1.0,
    high: float = 99.8,
    axis: tuple[int, ...] | None = None,
    clip: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Scale ``img`` so the ``low``/``high`` percentiles map to 0/1.

    For ``cyx`` input pass ``axis=(1, 2)`` to normalise each channel independently.
    """
    img = img.astype(np.float32, copy=False)
    lo = np.percentile(img, low, axis=axis, keepdims=axis is not None)
    hi = np.percentile(img, high, axis=axis, keepdims=axis is not None)
    out = (img - lo) / (hi - lo + eps)
    return np.clip(out, 0.0, 1.0) if clip else out


def to_uint8(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    return (percentile_normalize(img, low, high) * 255).astype(np.uint8)


def robust_zscore(x: np.ndarray, ref: np.ndarray | None = None, axis: int = 0) -> np.ndarray:
    """Median/MAD z-score, optionally relative to a reference population (e.g. DMSO wells)."""
    ref = x if ref is None else ref
    med = np.nanmedian(ref, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(ref - med), axis=axis, keepdims=True) * 1.4826
    return (x - med) / np.where(mad == 0, 1.0, mad)
