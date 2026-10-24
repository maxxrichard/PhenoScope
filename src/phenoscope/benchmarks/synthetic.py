"""Synthetic data with exact ground truth, for tests, demos and CI smoke benchmarks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import ellipse


def synthetic_cells(
    shape: tuple[int, int] = (256, 256),
    n_cells: int = 40,
    radius: tuple[float, float] = (6, 11),
    noise: float = 0.05,
    stain_gain: float = 1.0,
    background: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-channel (nucleus, cytoplasm) fluorescence image ``(2, y, x)`` + nucleus labels.

    ``stain_gain``/``background`` simulate staining-batch variation for drift tests.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    labels = np.zeros(shape, np.int32)
    cyto = np.zeros(shape, np.float32)
    placed = 0
    for _ in range(n_cells * 20):
        if placed >= n_cells:
            break
        r = rng.uniform(*radius)
        cy, cx = rng.uniform(r + 2, h - r - 2), rng.uniform(r + 2, w - r - 2)
        rr, cc = ellipse(cy, cx, r, r * rng.uniform(0.75, 1.0), shape, rotation=rng.uniform(0, np.pi))
        if labels[rr, cc].any() or (ndi.binary_dilation(labels > 0, iterations=2)[rr, cc]).any():
            continue
        placed += 1
        labels[rr, cc] = placed
        rr2, cc2 = ellipse(cy, cx, r * 2.0, r * 1.7, shape, rotation=rng.uniform(0, np.pi))
        cyto[rr2, cc2] += rng.uniform(0.3, 0.6)

    nuc = (labels > 0).astype(np.float32) * rng.uniform(0.7, 1.0, labels.max() + 1)[labels]
    nuc = ndi.gaussian_filter(nuc, 1.0)
    cyto = ndi.gaussian_filter(cyto, 2.0)
    img = np.stack([nuc, cyto]) * stain_gain + background
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    img = np.clip(img, 0, None)
    return (img * 4000).astype(np.uint16), labels


def synthetic_he(shape: tuple[int, int] = (512, 512), seed: int = 0) -> np.ndarray:
    """Crude H&E-like RGB image: pink stroma blob with purple nuclei on white glass."""
    rng = np.random.default_rng(seed)
    h, w = shape
    img = np.full((h, w, 3), 242, np.float32)
    yy, xx = np.mgrid[:h, :w]
    tissue = ((yy - h / 2) ** 2 / (h / 2.6) ** 2 + (xx - w / 2) ** 2 / (w / 2.4) ** 2) < 1
    img[tissue] = (225, 150, 190)
    _, nuc = synthetic_cells(shape, n_cells=int(h * w / 2500), radius=(3, 6), seed=seed)
    img[(nuc > 0) & tissue] = (90, 50, 140)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_multiplex(
    n_cycles: int = 4,
    markers_per_cycle: int = 3,
    shape: tuple[int, int] = (256, 256),
    max_shift: float = 12.0,
    seed: int = 0,
):
    """Cyclic multiplex stack with known per-cycle translations.

    Returns ``(cycles, shifts)`` where ``cycles`` is a list of
    :class:`~phenoscope.registration.CycleImage` (DAPI + markers) and ``shifts``
    the true ``(dy, dx)`` applied to each cycle.
    """
    from phenoscope.registration.multiplex import CycleImage

    rng = np.random.default_rng(seed)
    base, labels = synthetic_cells(shape, n_cells=60, seed=seed)
    dapi = base[0].astype(np.float32)
    cycles, shifts = [], []
    marker_id = 0
    for c in range(n_cycles):
        shift = (0.0, 0.0) if c == 0 else tuple(rng.uniform(-max_shift, max_shift, 2))
        shifts.append(shift)
        planes, names = [ndi.shift(dapi, shift, order=1)], ["DAPI"]
        for _ in range(markers_per_cycle):
            positive = rng.random(labels.max() + 1) < 0.4
            positive[0] = False
            marker = ndi.gaussian_filter(positive[labels].astype(np.float32) * 3000, 1.5)
            planes.append(ndi.shift(marker, shift, order=1))
            names.append(f"M{marker_id:02d}")
            marker_id += 1
        cycles.append(CycleImage(np.stack(planes), names, "DAPI"))
    return cycles, shifts
