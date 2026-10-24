"""Units of work executed by job backends (locally, or inside a GPU pod via ``phenoscope run-task``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from phenoscope.io.omezarr import read_labels, read_ome_zarr, write_labels
from phenoscope.io.storage import get_store
from phenoscope.segmentation import get_segmenter

TaskFn = Callable[[dict[str, Any]], dict[str, Any]]
TASKS: dict[str, TaskFn] = {}


def task(name: str) -> Callable[[TaskFn], TaskFn]:
    def deco(fn: TaskFn) -> TaskFn:
        TASKS[name] = fn
        return fn

    return deco


@task("segment")
def segment_task(p: dict[str, Any]) -> dict[str, Any]:
    """Payload: ``uri``, ``segmenter``, ``params``, ``channels`` (names), ``label_name``."""
    img = read_ome_zarr(p["uri"], level=p.get("level", 0))
    data = img.data
    if p.get("channels"):
        data = np.stack([img.channel(c) for c in p["channels"]])
    seg = get_segmenter(p.get("segmenter", "watershed"), **p.get("params", {}))
    res = seg.run(data)
    write_labels(res.labels, p["uri"], name=p.get("label_name", "cells"))
    return {"n_objects": res.n_objects, "runtime_s": res.runtime_s, "model": res.model,
            "params": res.params}


@task("features")
def features_task(p: dict[str, Any]) -> dict[str, Any]:
    """Payload: ``uri``, ``label_name``, ``out`` (parquet/csv URI), ``texture``."""
    from phenoscope.profiling.features import extract_features

    img = read_ome_zarr(p["uri"])
    labels = read_labels(p["uri"], p.get("label_name", "cells"))
    px = img.pixel_size_um[-1] if img.pixel_size_um else 1.0
    df = extract_features(img.data, labels, img.channel_names, px, texture=p.get("texture", False))
    out = p.get("out")
    if out:
        _write_table(df, out)
    return {"n_objects": len(df), "n_features": df.shape[1], "out": out}


@task("embed")
def embed_task(p: dict[str, Any]) -> dict[str, Any]:
    """Payload: ``uri``, ``label_name``, ``out`` (.npy URI), ``crop_size``, ``model``."""
    from phenoscope.profiling.embeddings import ViTEmbedder, extract_object_crops

    img = read_ome_zarr(p["uri"])
    labels = read_labels(p["uri"], p.get("label_name", "cells"))
    crops, ids = extract_object_crops(img.data, labels, size=p.get("crop_size", 96))
    emb = ViTEmbedder(p.get("model")).embed(crops)
    if p.get("out"):
        store = get_store(p["out"].rsplit("/", 1)[0], mode="w")
        import io

        buf = io.BytesIO()
        np.savez_compressed(buf, embeddings=emb, labels=ids)
        store[p["out"].rsplit("/", 1)[1]] = buf.getvalue()
    return {"n_objects": int(len(ids)), "dim": int(emb.shape[1]) if emb.size else 0}


def _write_table(df: pd.DataFrame, uri: str) -> None:
    parent, name = uri.rsplit("/", 1)
    store = get_store(parent, mode="w")
    store[name] = df.to_csv(index=False).encode() if name.endswith(".csv") else df.to_parquet()


def run_task(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind not in TASKS:
        raise KeyError(f"unknown task {kind!r}; available: {sorted(TASKS)}")
    return TASKS[kind](payload)
