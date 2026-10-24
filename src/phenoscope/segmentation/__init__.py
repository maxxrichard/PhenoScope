"""Cell, nucleus and tissue segmentation.

All segmenters share one interface (:class:`Segmenter`) and are looked up by name,
so pipelines, the API and the napari widget can switch backends via config::

    seg = get_segmenter("cellpose", model_type="cyto3", diameter=30)
    labels = seg.segment(image_cyx, channel_idx=(0, 1))
"""

# Importing the backend modules registers them; heavy deps (torch, cellpose,
# segment-anything, monai) are only imported when a model is first used.
from phenoscope.segmentation import cellpose_seg, sam_seg, unet, watershed  # noqa: F401
from phenoscope.segmentation.base import (
    SegmentationResult,
    Segmenter,
    available_segmenters,
    get_segmenter,
    register_segmenter,
)
from phenoscope.segmentation.postprocess import clean_labels, masks_to_labels

__all__ = [
    "SegmentationResult",
    "Segmenter",
    "available_segmenters",
    "clean_labels",
    "get_segmenter",
    "masks_to_labels",
    "register_segmenter",
]
