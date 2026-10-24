"""Retrospective flat-field correction for high-content screens.

A per-plate, per-channel flat-field is estimated from many fields of view: the
pixel-wise median across fields removes cellular content, and a wide Gaussian
smooths what remains into the illumination profile (CellProfiler-style
"median + smoothing" correction).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def estimate_flatfield(
    fields: np.ndarray, sigma: float = 50.0, downsample: int = 4
) -> np.ndarray:
    """Estimate a flat-field from a stack of fields ``(n, y, x)``.

    Returns a profile with mean 1 that can be divided out of each image.
    """
    if fields.ndim != 3:
        raise ValueError("expected (n_fields, y, x)")
    small = fields[:, ::downsample, ::downsample].astype(np.float32)
    profile = np.median(small, axis=0)
    profile = ndi.gaussian_filter(profile, sigma=sigma / downsample)
    zoom = (fields.shape[1] / profile.shape[0], fields.shape[2] / profile.shape[1])
    profile = ndi.zoom(profile, zoom, order=1)[: fields.shape[1], : fields.shape[2]]
    profile /= profile.mean() + 1e-8
    return np.clip(profile, 1e-3, None)


def correct_illumination(
    img: np.ndarray, flatfield: np.ndarray, darkfield: np.ndarray | float = 0.0
) -> np.ndarray:
    """Apply ``(img - darkfield) / flatfield``, preserving the input dtype range."""
    out = (img.astype(np.float32) - darkfield) / flatfield
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(out, info.min, info.max).astype(img.dtype)
    return out
