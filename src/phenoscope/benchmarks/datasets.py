"""Curated benchmark datasets.

Each benchmark is a frozen set of images with instance ground truth plus the
reference feature/embedding distribution used by the drift monitor. Specs live
in ``configs/benchmarks.yaml``; the data itself sits in OME-Zarr stores (local or
S3) with masks under ``labels/<label_name>``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from phenoscope.io.omezarr import read_labels, read_ome_zarr
from phenoscope.io.storage import join_uri


@dataclass
class BenchmarkSpec:
    name: str
    modality: str  # fluorescence | brightfield | he | multiplex
    uri: str
    images: list[str]
    label_name: str = "cells"
    channels: list[str] = field(default_factory=list)
    split: dict[str, list[str]] = field(default_factory=dict)
    description: str = ""


def load_registry(path: str | Path = "configs/benchmarks.yaml") -> dict[str, BenchmarkSpec]:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return {name: BenchmarkSpec(name=name, **spec) for name, spec in raw["benchmarks"].items()}


class InstanceDataset(Sequence):
    """``(image_cyx, labels_yx)`` pairs for a benchmark split, loaded lazily."""

    def __init__(self, spec: BenchmarkSpec, split: str | None = None) -> None:
        self.spec = spec
        self.ids = spec.split.get(split, spec.images) if split else spec.images

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):  # type: ignore[override]
        uri = join_uri(self.spec.uri, f"{self.ids[i]}.zarr")
        img = read_ome_zarr(uri)
        data = img.data
        if self.spec.channels:
            data = np.stack([img.channel(c) for c in self.spec.channels])
        return data, read_labels(uri, self.spec.label_name)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]
