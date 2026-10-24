"""Segment Anything (SAM) backend with support for fine-tuned mask-decoder weights.

The vanilla SAM automatic mask generator over-segments texture in microscopy;
fine-tuning the mask decoder on curated instance masks (see
:mod:`phenoscope.segmentation.training`) and filtering by stability/area makes it
competitive with Cellpose on dense fluorescence and H&E nuclei.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phenoscope.config import get_settings, resolve_device
from phenoscope.preprocessing.normalize import to_uint8
from phenoscope.segmentation.base import Segmenter, register_segmenter
from phenoscope.segmentation.postprocess import masks_to_labels


def load_sam(model_type: str, checkpoint: str | Path, decoder_weights: str | Path | None = None):
    try:
        import torch
        from segment_anything import sam_model_registry
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("SAM backend needs `pip install phenoscope[ml]`") from exc
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    if decoder_weights:
        state = torch.load(decoder_weights, map_location="cpu")
        sam.mask_decoder.load_state_dict(state)
    return sam.to(resolve_device(get_settings().device)).eval()


@register_segmenter("sam")
class SAMSegmenter(Segmenter):
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: str | None = None,
        decoder_weights: str | None = None,
        points_per_side: int = 48,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.90,
        max_area_frac: float = 0.05,
        min_size: int = 30,
        **params,
    ) -> None:
        super().__init__(
            min_size=min_size,
            model_type=model_type,
            decoder_weights=decoder_weights,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            **params,
        )
        self.model_type = model_type
        self.checkpoint = checkpoint or str(get_settings().model_path / f"sam_{model_type}.pth")
        self.decoder_weights = decoder_weights
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.max_area_frac = max_area_frac
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            from segment_anything import SamAutomaticMaskGenerator

            sam = load_sam(self.model_type, self.checkpoint, self.decoder_weights)
            self._generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_size,
            )
        return self._generator

    @staticmethod
    def to_rgb(image: np.ndarray) -> np.ndarray:
        """Map a ``cyx`` image to the 8-bit RGB input SAM expects."""
        if image.shape[0] >= 3:
            return np.stack([to_uint8(image[i]) for i in range(3)], axis=-1)
        chans = [to_uint8(image[0])] * 3
        if image.shape[0] == 2:
            chans[1] = to_uint8(image[1])
        return np.stack(chans, axis=-1)

    def _segment(self, image: np.ndarray) -> np.ndarray:
        rgb = self.to_rgb(image)
        max_area = self.max_area_frac * rgb.shape[0] * rgb.shape[1]
        anns = [a for a in self.generator.generate(rgb) if a["area"] <= max_area]
        masks = [a["segmentation"] for a in anns]
        scores = [a["predicted_iou"] * a["stability_score"] for a in anns]
        return masks_to_labels(masks, scores, shape=rgb.shape[:2])
