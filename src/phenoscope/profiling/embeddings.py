"""Learned phenotypic embeddings from a Vision Transformer.

Single-cell crops (fluorescence / brightfield) or tissue tiles (H&E WSI) are fed
to a self-supervised ViT (DINO / DINOv2 by default via ``timm``); the CLS token
is the morphological profile used for clustering, MoA prediction and fusion with
transcriptomics.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from skimage.measure import regionprops
from skimage.transform import resize

from phenoscope.config import get_settings, resolve_device
from phenoscope.preprocessing.normalize import percentile_normalize


def extract_object_crops(
    image: np.ndarray, labels: np.ndarray, size: int = 96, mask_background: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-size crops centred on each object. Returns ``(crops (n, c, s, s), label_ids)``."""
    if image.ndim == 2:
        image = image[None]
    half = size // 2
    padded = np.pad(image, ((0, 0), (half, half), (half, half)), mode="constant")
    lpad = np.pad(labels, ((half, half), (half, half)), mode="constant")
    crops, ids = [], []
    for r in regionprops(labels):
        cy, cx = (int(round(v)) + half for v in r.centroid)
        crop = padded[:, cy - half : cy + half, cx - half : cx + half]
        if mask_background:
            crop = crop * (lpad[cy - half : cy + half, cx - half : cx + half] == r.label)
        crops.append(crop)
        ids.append(r.label)
    if not crops:
        return np.zeros((0, image.shape[0], size, size), image.dtype), np.zeros(0, int)
    return np.stack(crops), np.asarray(ids)


def _batches(arr: np.ndarray, n: int) -> Iterator[np.ndarray]:
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


class ViTEmbedder:
    """Wraps a ``timm`` ViT. Multi-channel inputs are embedded per channel and concatenated
    (channel-agnostic, as in Cell Painting work), or mapped to RGB for H&E."""

    def __init__(
        self,
        model_name: str | None = None,
        input_size: int = 224,
        batch_size: int = 128,
        per_channel: bool = True,
        checkpoint: str | None = None,
    ) -> None:
        self.model_name = model_name or get_settings().vit_model
        self.input_size = input_size
        self.batch_size = batch_size
        self.per_channel = per_channel
        self.checkpoint = checkpoint
        self._model = None
        self.device = resolve_device(get_settings().device)

    @property
    def model(self):
        if self._model is None:
            try:
                import timm
                import torch
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("ViT embeddings need `pip install phenoscope[ml]`") from exc
            model = timm.create_model(
                self.model_name, pretrained=self.checkpoint is None, num_classes=0
            )
            if self.checkpoint:
                model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"), strict=False)
            self._model = model.to(self.device).eval()
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.num_features)

    def _prepare(self, crops: np.ndarray) -> np.ndarray:
        """``(n, 3, s, s)`` float32 at input size with ImageNet normalisation."""
        n, c, s, _ = crops.shape
        x = percentile_normalize(crops, axis=(2, 3))
        if s != self.input_size:
            x = np.stack([resize(img, (c, self.input_size, self.input_size), order=1,
                                 anti_aliasing=True) for img in x])
        mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
        return ((x - mean) / std).astype(np.float32)

    def embed(self, crops: np.ndarray) -> np.ndarray:
        """Embed ``(n, c, s, s)`` crops → ``(n, d)`` (or ``(n, c*d)`` when ``per_channel``)."""
        import torch

        if len(crops) == 0:
            return np.zeros((0, self.dim), np.float32)
        if self.per_channel and crops.shape[1] != 3:
            per = [self.embed(np.repeat(crops[:, i : i + 1], 3, axis=1)) for i in range(crops.shape[1])]
            return np.concatenate(per, axis=1)
        if crops.shape[1] == 1:
            crops = np.repeat(crops, 3, axis=1)
        out = []
        with torch.no_grad():
            for batch in _batches(self._prepare(crops[:, :3]), self.batch_size):
                feats = self.model(torch.from_numpy(batch).to(self.device))
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out)

    def embed_stream(self, crops: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate([self.embed(c) for c in crops])
