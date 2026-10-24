"""Batch worker: ``python -m phenoscope.pipelines.batch '<json list of URIs>'``.

Task and parameters come from ``PHENOSCOPE_TASK`` / ``PHENOSCOPE_TASK_PARAMS``
(set by the Airflow ``KubernetesPodOperator``). Models are loaded once per pod and
reused across the batch, which is what makes GPU pods worthwhile.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd

from phenoscope.io.omezarr import read_labels, read_ome_zarr, write_labels
from phenoscope.io.storage import get_store
from phenoscope.profiling.aggregate import aggregate_profiles
from phenoscope.segmentation import get_segmenter

log = logging.getLogger("phenoscope.batch")


def segment_batch(uris: list[str], segmenter: str = "cellpose", label_name: str = "cells",
                  **params) -> list[dict]:
    seg = get_segmenter(segmenter, **params)
    out = []
    for uri in uris:
        res = seg.run(read_ome_zarr(uri).data)
        write_labels(res.labels, uri, name=label_name)
        out.append({"uri": uri, "n_objects": res.n_objects, "runtime_s": res.runtime_s})
        log.info("%s: %d objects in %.1fs", uri, res.n_objects, res.runtime_s)
    return out


def features_embed_batch(uris: list[str], label_name: str = "cells", embed: bool = True) -> None:
    import numpy as np

    from phenoscope.profiling.embeddings import ViTEmbedder, extract_object_crops
    from phenoscope.profiling.features import extract_features

    embedder = ViTEmbedder() if embed else None
    for uri in uris:
        img = read_ome_zarr(uri)
        labels = read_labels(uri, label_name)
        df = extract_features(img.data, labels, img.channel_names, img.pixel_size_um[-1])
        store = get_store(uri, mode="a")
        store["profiles/features.parquet"] = df.to_parquet()
        if embedder is not None:
            crops, ids = extract_object_crops(img.data, labels)
            emb = embedder.embed(crops)
            import io

            buf = io.BytesIO()
            np.savez_compressed(buf, embeddings=emb, labels=ids)
            store["profiles/vit.npz"] = buf.getvalue()


def aggregate_run(uris: list[str]) -> str:
    """Merge per-image features into one per-well profile table next to the first image."""
    import io

    frames = []
    for uri in uris:
        raw = get_store(uri)["profiles/features.parquet"]
        df = pd.read_parquet(io.BytesIO(raw))
        stem = uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")
        plate, _, well = stem.partition("_")
        frames.append(df.assign(plate=plate, well=well or stem))
    profiles = aggregate_profiles(pd.concat(frames, ignore_index=True))
    dest = uris[0].rsplit("/", 1)[0] + "/_profiles"
    get_store(dest, mode="w")["well_profiles.parquet"] = profiles.to_parquet()
    return dest + "/well_profiles.parquet"


def main(argv: list[str]) -> None:
    logging.basicConfig(level=logging.INFO)
    uris = json.loads(argv[0])
    task = os.environ.get("PHENOSCOPE_TASK", "segment")
    params = json.loads(os.environ.get("PHENOSCOPE_TASK_PARAMS", "{}"))
    if task == "segment":
        segment_batch(uris, **params)
    elif task == "features+embed":
        features_embed_batch(uris, **params)
    else:
        raise SystemExit(f"unknown task {task}")


if __name__ == "__main__":
    main(sys.argv[1:])
