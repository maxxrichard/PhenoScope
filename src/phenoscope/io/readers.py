"""Readers for common acquisition formats, normalised to ``cyx`` numpy arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from skimage import io as skio


def _to_cyx(arr: np.ndarray, rgb: bool) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3 and (rgb or arr.shape[-1] in (3, 4)) and arr.shape[0] not in (3, 4):
        return np.moveaxis(arr[..., :3], -1, 0)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"unsupported image shape {arr.shape}; expected 2D or 3D")


def read_image(path: str | Path, rgb: bool = False) -> np.ndarray:
    """Read TIFF/OME-TIFF/PNG/JPEG into a ``(c, y, x)`` array.

    Multi-page TIFFs are interpreted as channel stacks. Whole-slide formats
    (.svs, .ndpi, .mrxs) go through :mod:`phenoscope.wsi` instead.
    """
    path = Path(path)
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith((".tif", ".tiff")):
        arr = tifffile.imread(path)
        if arr.ndim == 4:  # e.g. (z, c, y, x) -> max-project z
            arr = arr.max(axis=0)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = skio.imread(path)
    return _to_cyx(np.asarray(arr), rgb=rgb)
