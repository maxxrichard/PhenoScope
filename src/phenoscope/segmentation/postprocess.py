"""Label-image post-processing shared by all segmenters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import relabel_sequential


def clean_labels(
    labels: np.ndarray, min_size: int = 30, fill_holes: bool = True, max_size: int | None = None
) -> np.ndarray:
    """Remove small/huge objects, fill holes and relabel to ``1..N``."""
    labels = np.asarray(labels).astype(np.int32, copy=False)
    if labels.max() == 0:
        return labels
    sizes = np.bincount(labels.ravel())
    drop = sizes < min_size
    if max_size is not None:
        drop |= sizes > max_size
    drop[0] = False
    if drop.any():
        labels = labels.copy()
        labels[drop[labels]] = 0
    if fill_holes:
        filled = labels.copy()
        for idx, sl in enumerate(ndi.find_objects(labels), start=1):
            if sl is None:
                continue
            obj = labels[sl] == idx
            holes = ndi.binary_fill_holes(obj) & ~obj & (labels[sl] == 0)
            filled[sl][holes] = idx
        labels = filled
    return relabel_sequential(labels)[0].astype(np.int32)


def masks_to_labels(
    masks: Sequence[np.ndarray], scores: Sequence[float] | None = None, shape=None
) -> np.ndarray:
    """Merge (possibly overlapping) boolean masks into one label image.

    Higher-scoring masks are painted last so they win overlaps; masks that lose more
    than half their area to better masks are dropped.
    """
    if len(masks) == 0:
        return np.zeros(shape, dtype=np.int32)
    shape = shape or masks[0].shape
    order = np.argsort(scores) if scores is not None else np.arange(len(masks))
    labels = np.zeros(shape, dtype=np.int32)
    for new_id, i in enumerate(order, start=1):
        labels[masks[i].astype(bool)] = new_id
    # Drop masks mostly overwritten by better-scoring ones.
    areas = np.bincount(labels.ravel(), minlength=len(masks) + 1)
    for new_id, i in enumerate(order, start=1):
        if areas[new_id] < 0.5 * masks[i].sum():
            labels[labels == new_id] = 0
    return relabel_sequential(labels)[0].astype(np.int32)
