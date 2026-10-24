from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from skimage.segmentation import find_boundaries

from phenoscope.api.schemas import ImageOut, JobOut, SegmentationOut, SegmentRequest
from phenoscope.db.models import Image, SegmentationRun
from phenoscope.db.session import get_session
from phenoscope.io.omezarr import list_levels, read_labels, read_ome_zarr
from phenoscope.jobs.backends import get_backend
from phenoscope.jobs.tasks import run_task
from phenoscope.preprocessing.normalize import to_uint8

router = APIRouter(prefix="/images", tags=["images"])

_COLORS = np.array([[0, 90, 255], [0, 220, 90], [255, 60, 60], [255, 0, 255], [255, 220, 0]])


def _get_image(image_id: int) -> Image:
    with get_session() as s:
        img = s.get(Image, image_id)
        if img is None:
            raise HTTPException(404, "image not found")
        return img


@router.get("/{image_id}", response_model=ImageOut)
def get_image(image_id: int) -> ImageOut:
    return ImageOut.model_validate(_get_image(image_id))


@router.get("/{image_id}/render.png", responses={200: {"content": {"image/png": {}}}})
def render(
    image_id: int,
    level: int | None = Query(None, ge=0),
    channels: str | None = Query(None, description="comma-separated channel names"),
    labels: str | None = Query(None, description="overlay outlines of this label set"),
) -> Response:
    """Composite PNG preview used by the web viewer (additive colour blend + outlines)."""
    import cv2

    row = _get_image(image_id)
    if level is None:
        # Coarsest level that still has >= 1024 px on its long side (level 0 for small images).
        level = 0
        while level + 1 < list_levels(row.uri) and max(row.width, row.height) >> (level + 1) >= 1024:
            level += 1
    img = read_ome_zarr(row.uri, level=level)
    names = channels.split(",") if channels else img.channel_names[:3]
    rgb = np.zeros((*img.channel(names[0]).shape[-2:], 3), np.float32)
    for i, name in enumerate(names):
        rgb += to_uint8(img.channel(name))[..., None].astype(np.float32) * _COLORS[i % 5] / 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if labels:
        lab = read_labels(row.uri, labels, level=level)
        rgb[find_boundaries(lab[: rgb.shape[0], : rgb.shape[1]])] = (255, 255, 255)
    ok, buf = cv2.imencode(".png", rgb[..., ::-1])
    return Response(io.BytesIO(buf).getvalue(), media_type="image/png")


@router.post("/{image_id}/segment", response_model=SegmentationOut | JobOut)
def segment(image_id: int, body: SegmentRequest):
    row = _get_image(image_id)
    payload = {"uri": row.uri, "segmenter": body.segmenter, "params": body.params,
               "channels": body.channels, "label_name": body.label_name, "image_id": image_id}
    if body.asynchronous:
        backend = get_backend()
        return JobOut(**backend.status(backend.submit("segment", payload)))
    try:
        result = run_task("segment", payload)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    with get_session() as s:
        run = SegmentationRun(image_id=image_id, model=result["model"], params=result["params"],
                              label_name=body.label_name, n_objects=result["n_objects"],
                              runtime_s=result["runtime_s"])
        s.add(run)
        s.flush()
        return SegmentationOut.model_validate(run)


@router.get("/{image_id}/segmentations", response_model=list[SegmentationOut])
def segmentations(image_id: int):
    with get_session() as s:
        img = s.get(Image, image_id)
        if img is None:
            raise HTTPException(404, "image not found")
        return [SegmentationOut.model_validate(r) for r in img.segmentations]


@router.get("/{image_id}/features")
def features(image_id: int, label_name: str = "cells", limit: int = Query(1000, le=100_000)):
    row = _get_image(image_id)
    from phenoscope.profiling.features import extract_features

    img = read_ome_zarr(row.uri)
    try:
        lab = read_labels(row.uri, label_name)
    except KeyError as exc:
        raise HTTPException(404, f"no labels {label_name!r}; run segmentation first") from exc
    df = extract_features(img.data, lab, img.channel_names, row.pixel_size_um)
    return {"n_objects": len(df), "columns": list(df.columns),
            "rows": df.head(limit).to_dict(orient="records")}
