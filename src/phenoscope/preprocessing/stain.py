"""Macenko stain normalisation for H&E whole-slide tiles.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis", ISBI 2009.
"""

from __future__ import annotations

import numpy as np

# Default H&E stain matrix and max concentrations (from the original paper).
_REF_STAINS = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
_REF_MAX_C = np.array([1.9705, 1.0308])


def rgb_to_od(rgb: np.ndarray, io: float = 240.0) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return -np.log((rgb + 1.0) / io)


def od_to_rgb(od: np.ndarray, io: float = 240.0) -> np.ndarray:
    return np.clip(io * np.exp(-od), 0, 255).astype(np.uint8)


def estimate_stain_matrix(
    rgb: np.ndarray, beta: float = 0.15, alpha: float = 1.0, io: float = 240.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(stain_matrix (3x2), max_concentrations (2,))`` for an RGB tile ``(y, x, 3)``."""
    od = rgb_to_od(rgb.reshape(-1, 3), io)
    od_hat = od[np.all(od > beta, axis=1)]
    if od_hat.shape[0] < 10:
        raise ValueError("too few tissue pixels to estimate stain vectors")

    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    plane = eigvecs[:, 1:3]
    proj = od_hat @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi, max_phi = np.percentile(phi, [alpha, 100 - alpha])
    v_min = plane @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v_max = plane @ np.array([np.cos(max_phi), np.sin(max_phi)])
    # Haematoxylin is the vector with the larger red OD component.
    stains = np.array([v_min, v_max]).T if v_min[0] > v_max[0] else np.array([v_max, v_min]).T
    stains = stains * np.sign(stains.sum(axis=0, keepdims=True))

    conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
    max_c = np.percentile(conc, 99, axis=1)
    return stains, max_c


class MacenkoNormalizer:
    """Fit on a reference tile, then map other tiles into its stain space."""

    def __init__(self, io: float = 240.0) -> None:
        self.io = io
        self.stains = _REF_STAINS.copy()
        self.max_c = _REF_MAX_C.copy()

    def fit(self, reference_rgb: np.ndarray) -> MacenkoNormalizer:
        self.stains, self.max_c = estimate_stain_matrix(reference_rgb, io=self.io)
        return self

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        h, w, _ = rgb.shape
        stains, max_c = estimate_stain_matrix(rgb, io=self.io)
        od = rgb_to_od(rgb.reshape(-1, 3), self.io)
        conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
        conc *= (self.max_c / np.maximum(max_c, 1e-6))[:, None]
        return od_to_rgb((self.stains @ conc).T, self.io).reshape(h, w, 3)

    def separate(self, rgb: np.ndarray) -> np.ndarray:
        """Return per-pixel stain concentrations ``(2, y, x)`` (H, E)."""
        h, w, _ = rgb.shape
        od = rgb_to_od(rgb.reshape(-1, 3), self.io)
        conc = np.linalg.lstsq(self.stains, od.T, rcond=None)[0]
        return conc.reshape(2, h, w)
