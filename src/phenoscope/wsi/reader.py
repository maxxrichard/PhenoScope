"""Uniform slide interface over OpenSlide (.svs/.ndpi/.mrxs) and in-memory arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from skimage.transform import downscale_local_mean


class Slide(Protocol):
    dimensions: tuple[int, int]  # (width, height) at level 0
    mpp: float

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0) -> np.ndarray: ...
    def thumbnail(self, downsample: int) -> np.ndarray: ...


class ArraySlide:
    """Wrap an RGB ``(y, x, 3)`` array so tests and OME-Zarr H&E use the same code path."""

    def __init__(self, rgb: np.ndarray, mpp: float = 0.5) -> None:
        self.rgb = rgb
        self.mpp = mpp
        self.dimensions = (rgb.shape[1], rgb.shape[0])

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0) -> np.ndarray:
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        crop = self.rgb[y : y + h, x : x + w]
        out[: crop.shape[0], : crop.shape[1]] = crop
        return out

    def thumbnail(self, downsample: int) -> np.ndarray:
        return downscale_local_mean(self.rgb, (downsample, downsample, 1)).astype(np.uint8)


class OpenSlideSlide:
    def __init__(self, path: str | Path) -> None:
        try:
            import openslide
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("WSI support needs `pip install phenoscope[wsi]`") from exc
        self._s = openslide.OpenSlide(str(path))
        self.dimensions = self._s.dimensions
        self.mpp = float(self._s.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))

    def read_region(self, x: int, y: int, w: int, h: int, level: int = 0) -> np.ndarray:
        return np.asarray(self._s.read_region((x, y), level, (w, h)).convert("RGB"))

    def thumbnail(self, downsample: int) -> np.ndarray:
        w, h = self.dimensions
        return np.asarray(self._s.get_thumbnail((w // downsample, h // downsample)).convert("RGB"))


def open_slide(path: str | Path) -> Slide:
    path = Path(path)
    if path.suffix.lower() in {".npy"}:
        return ArraySlide(np.load(path))
    return OpenSlideSlide(path)
