"""MONAI U-Net with a 3-class (background / interior / boundary) head.

Instances are recovered from the interior probability with a seeded watershed,
which handles touching nuclei in tissue far better than plain thresholding.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries, watershed

from phenoscope.config import get_settings, resolve_device
from phenoscope.preprocessing.normalize import percentile_normalize
from phenoscope.segmentation.base import Segmenter, register_segmenter


def build_unet(in_channels: int = 1, out_channels: int = 3):
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("U-Net backend needs `pip install phenoscope[ml]`") from exc
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
        dropout=0.1,
    )


def labels_to_three_class(labels: np.ndarray) -> np.ndarray:
    """Training target: 0 background, 1 interior, 2 boundary."""
    target = (labels > 0).astype(np.int64)
    target[find_boundaries(labels, mode="inner")] = 2
    return target


def three_class_to_labels(
    probs: np.ndarray, interior_thresh: float = 0.5, seed_thresh: float = 0.8
) -> np.ndarray:
    """Convert softmax output ``(3, y, x)`` to an instance label image."""
    interior = probs[1]
    foreground = (probs[1] + probs[2]) > interior_thresh
    seeds = ndi.label(interior > seed_thresh)[0]
    return watershed(-interior, seeds, mask=foreground).astype(np.int32)


@register_segmenter("unet")
class UNetSegmenter(Segmenter):
    def __init__(
        self,
        weights: str | None = None,
        in_channels: int = 1,
        roi_size: int = 256,
        min_size: int = 30,
        **params,
    ) -> None:
        super().__init__(min_size=min_size, weights=weights, roi_size=roi_size, **params)
        self.weights = weights or str(get_settings().model_path / "unet_nuclei.pt")
        self.in_channels = in_channels
        self.roi_size = roi_size
        self._net = None

    @property
    def net(self):
        if self._net is None:
            import torch

            net = build_unet(self.in_channels)
            net.load_state_dict(torch.load(self.weights, map_location="cpu"))
            self._net = net.to(resolve_device(get_settings().device)).eval()
        return self._net

    def _segment(self, image: np.ndarray) -> np.ndarray:
        import torch
        from monai.inferers import sliding_window_inference

        x = percentile_normalize(image[: self.in_channels], axis=(1, 2))
        device = next(self.net.parameters()).device
        with torch.no_grad():
            t = torch.from_numpy(x)[None].to(device)
            logits = sliding_window_inference(
                t, (self.roi_size, self.roi_size), sw_batch_size=8, predictor=self.net, overlap=0.25
            )
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return three_class_to_labels(probs)
