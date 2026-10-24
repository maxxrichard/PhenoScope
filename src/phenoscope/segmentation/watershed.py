"""Classical seeded-watershed segmenter: CPU-only baseline and test fixture backend."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed

from phenoscope.preprocessing.normalize import percentile_normalize
from phenoscope.segmentation.base import Segmenter, register_segmenter


@register_segmenter("watershed")
class WatershedSegmenter(Segmenter):
    """Otsu foreground + distance-transform seeds + watershed.

    Uses the first channel (nuclear stain) for seeds; if a second channel is
    given, the watershed floods its intensity to approximate cell bodies.
    """

    def __init__(
        self, sigma: float = 1.5, min_distance: int = 6, min_size: int = 30, **params
    ) -> None:
        super().__init__(min_size=min_size, sigma=sigma, min_distance=min_distance, **params)
        self.sigma = sigma
        self.min_distance = min_distance

    def _segment(self, image: np.ndarray) -> np.ndarray:
        nuc = gaussian(percentile_normalize(image[0]), sigma=self.sigma)
        fg = nuc > threshold_otsu(nuc)
        fg = ndi.binary_opening(fg, iterations=1)
        dist = ndi.distance_transform_edt(fg)
        coords = peak_local_max(
            dist, min_distance=self.min_distance, labels=ndi.label(fg)[0], exclude_border=False
        )
        markers = np.zeros_like(dist, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

        if image.shape[0] > 1:
            cyto = gaussian(percentile_normalize(image[1]), sigma=self.sigma)
            mask = (cyto > threshold_otsu(cyto) * 0.5) | fg
            return watershed(-cyto, markers, mask=mask)
        return watershed(-dist, markers, mask=fg)
