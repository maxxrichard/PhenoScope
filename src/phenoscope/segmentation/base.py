"""Segmenter interface and registry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_REGISTRY: dict[str, type[Segmenter]] = {}


@dataclass
class SegmentationResult:
    labels: np.ndarray
    model: str
    runtime_s: float
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return int(len(np.unique(self.labels)) - (1 if (self.labels == 0).any() else 0))


class Segmenter(ABC):
    """Base class. Subclasses implement :meth:`_segment` on a ``(c, y, x)`` float image."""

    name: str = "base"

    def __init__(self, min_size: int = 30, **params: Any) -> None:
        self.min_size = min_size
        self.params = params

    @abstractmethod
    def _segment(self, image: np.ndarray) -> np.ndarray: ...

    def segment(
        self,
        image: np.ndarray,
        channel_idx: Sequence[int] | None = None,
    ) -> np.ndarray:
        return self.run(image, channel_idx).labels

    def run(self, image: np.ndarray, channel_idx: Sequence[int] | None = None) -> SegmentationResult:
        from phenoscope.segmentation.postprocess import clean_labels

        if image.ndim == 2:
            image = image[None]
        if channel_idx is not None:
            image = image[list(channel_idx)]
        t0 = time.perf_counter()
        labels = self._segment(image.astype(np.float32, copy=False))
        labels = clean_labels(labels, min_size=self.min_size)
        return SegmentationResult(
            labels=labels,
            model=self.name,
            runtime_s=time.perf_counter() - t0,
            params={"min_size": self.min_size, **self.params},
        )


def register_segmenter(name: str) -> Callable[[type[Segmenter]], type[Segmenter]]:
    def deco(cls: type[Segmenter]) -> type[Segmenter]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_segmenter(name: str, **kwargs: Any) -> Segmenter:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown segmenter {name!r}; available: {sorted(_REGISTRY)}") from exc
    return cls(**kwargs)


def available_segmenters() -> list[str]:
    return sorted(_REGISTRY)
