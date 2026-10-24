"""Tissue detection and tile extraction for H&E whole-slide images."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

from phenoscope.wsi.reader import Slide


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    size: int
    tissue_fraction: float


def tissue_mask(thumb_rgb: np.ndarray, min_area: int = 64) -> np.ndarray:
    """Otsu on HSV saturation, then morphology cleanup. Pen marks and glass are excluded."""
    hsv = rgb2hsv(thumb_rgb)
    sat = hsv[..., 1]
    if sat.max() - sat.min() < 1e-3:
        return np.zeros(sat.shape, bool)
    mask = sat > max(threshold_otsu(sat), 0.05)
    # Very dark pixels (pen ink, folds) are not tissue.
    mask &= hsv[..., 2] > 0.15
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


def tile_grid(
    slide: Slide,
    tile_size: int = 512,
    stride: int | None = None,
    min_tissue: float = 0.5,
    thumb_downsample: int = 32,
) -> list[Tile]:
    """Tiles at level 0 whose tissue fraction (estimated on the thumbnail) >= ``min_tissue``."""
    stride = stride or tile_size
    thumb = slide.thumbnail(thumb_downsample)
    mask = tissue_mask(thumb)
    w, h = slide.dimensions
    sx, sy = mask.shape[1] / w, mask.shape[0] / h
    tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            y0, y1 = int(y * sy), max(int((y + tile_size) * sy), int(y * sy) + 1)
            x0, x1 = int(x * sx), max(int((x + tile_size) * sx), int(x * sx) + 1)
            frac = float(mask[y0:y1, x0:x1].mean()) if mask[y0:y1, x0:x1].size else 0.0
            if frac >= min_tissue:
                tiles.append(Tile(x, y, tile_size, frac))
    return tiles


def iter_tiles(slide: Slide, tiles: list[Tile]) -> Iterator[tuple[Tile, np.ndarray]]:
    for t in tiles:
        yield t, slide.read_region(t.x, t.y, t.size, t.size)
