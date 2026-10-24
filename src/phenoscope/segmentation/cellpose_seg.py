"""Cellpose backend (generalist cyto/nuclei models or custom fine-tuned weights)."""

from __future__ import annotations

import numpy as np

from phenoscope.config import get_settings, resolve_device
from phenoscope.segmentation.base import Segmenter, register_segmenter


@register_segmenter("cellpose")
class CellposeSegmenter(Segmenter):
    def __init__(
        self,
        model_type: str = "cyto3",
        pretrained_model: str | None = None,
        diameter: float | None = 30.0,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 30,
        **params,
    ) -> None:
        super().__init__(
            min_size=min_size,
            model_type=model_type,
            pretrained_model=pretrained_model,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            **params,
        )
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self._model = None
        self._model_type = model_type
        self._pretrained = pretrained_model

    @property
    def model(self):
        if self._model is None:
            try:
                from cellpose import models
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("cellpose backend needs `pip install phenoscope[ml]`") from exc
            gpu = resolve_device(get_settings().device) == "cuda"
            if self._pretrained:
                self._model = models.CellposeModel(gpu=gpu, pretrained_model=self._pretrained)
            else:
                self._model = models.CellposeModel(gpu=gpu, model_type=self._model_type)
        return self._model

    def _segment(self, image: np.ndarray) -> np.ndarray:
        # Cellpose expects (y, x, c) with [cyto, nucleus] channel order; we pass
        # cyto as channel 1 when available, otherwise grayscale nucleus only.
        if image.shape[0] >= 2:
            img = np.stack([image[1], image[0]], axis=-1)
            channels = [1, 2]
        else:
            img = image[0]
            channels = [0, 0]
        out = self.model.eval(
            img,
            diameter=self.diameter,
            channels=channels,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
        )
        return np.asarray(out[0], dtype=np.int32)
