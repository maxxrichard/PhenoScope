"""Minimal, dependency-light OME-Zarr (NGFF v0.4) reader/writer.

Images are stored as multiscale pyramids with ``omero`` channel metadata and
optional ``labels/`` groups for segmentation masks, so the output opens directly
in napari (``napari-ome-zarr``), vizarr or QuPath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from skimage.transform import downscale_local_mean

from phenoscope.io.storage import get_store

NGFF_VERSION = "0.4"
_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


@dataclass
class OmeZarrImage:
    data: np.ndarray
    axes: str
    channel_names: list[str]
    pixel_size_um: tuple[float, ...]
    level: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.data.shape[self.axes.index("c")] if "c" in self.axes else 1

    def channel(self, name: str) -> np.ndarray:
        idx = self.channel_names.index(name)
        return np.take(self.data, idx, axis=self.axes.index("c"))


def _axes_meta(axes: str) -> list[dict[str, str]]:
    meta = []
    for ax in axes:
        if ax == "c":
            meta.append({"name": "c", "type": "channel"})
        elif ax == "t":
            meta.append({"name": "t", "type": "time", "unit": "second"})
        else:
            meta.append({"name": ax, "type": "space", "unit": "micrometer"})
    return meta


def _downsample(arr: np.ndarray, axes: str, factor: int = 2) -> np.ndarray:
    factors = tuple(factor if ax in "yx" else 1 for ax in axes)
    out = downscale_local_mean(arr, factors)
    return out.astype(arr.dtype)


def _build_pyramid(arr: np.ndarray, axes: str, levels: int, labels: bool) -> list[np.ndarray]:
    pyramid = [arr]
    for _ in range(1, levels):
        prev = pyramid[-1]
        if min(prev.shape[axes.index("y")], prev.shape[axes.index("x")]) < 64:
            break
        if labels:
            slices = tuple(slice(None, None, 2) if ax in "yx" else slice(None) for ax in axes)
            pyramid.append(prev[slices])
        else:
            pyramid.append(_downsample(prev, axes))
    return pyramid


def _multiscales(axes: str, n_levels: int, pixel_size_um: tuple[float, ...], name: str) -> dict:
    datasets = []
    for level in range(n_levels):
        scale = []
        spatial = iter(pixel_size_um)
        for ax in axes:
            if ax in "zyx":
                s = next(spatial, 1.0)
                scale.append(s * (2**level) if ax in "yx" else s)
            else:
                scale.append(1.0)
        datasets.append(
            {"path": str(level), "coordinateTransformations": [{"type": "scale", "scale": scale}]}
        )
    return {
        "version": NGFF_VERSION,
        "name": name,
        "axes": _axes_meta(axes),
        "datasets": datasets,
    }


def _omero(arr: np.ndarray, axes: str, channel_names: list[str]) -> dict:
    c_axis = axes.index("c") if "c" in axes else None
    channels = []
    palette = ["0000FF", "00FF00", "FF0000", "FF00FF", "FFFF00", "00FFFF", "FFFFFF"]
    for i, name in enumerate(channel_names):
        plane = np.take(arr, i, axis=c_axis) if c_axis is not None else arr
        lo, hi = np.percentile(plane, [0.5, 99.5]) if plane.size else (0, 1)
        channels.append(
            {
                "label": name,
                "color": palette[i % len(palette)],
                "active": i < 4,
                "window": {"start": float(lo), "end": float(hi), "min": float(plane.min()),
                           "max": float(plane.max())},
            }
        )
    return {"channels": channels, "rdefs": {"model": "color"}}


def write_ome_zarr(
    data: np.ndarray,
    uri: str,
    axes: str = "cyx",
    channel_names: list[str] | None = None,
    pixel_size_um: tuple[float, ...] = (1.0, 1.0),
    levels: int = 4,
    chunk: int = 512,
    name: str = "image",
    extra_attrs: dict[str, Any] | None = None,
) -> zarr.Group:
    """Write ``data`` as an NGFF multiscale image at ``uri`` (local path or s3://)."""
    if data.ndim != len(axes):
        raise ValueError(f"data has {data.ndim} dims but axes={axes!r}")
    n_c = data.shape[axes.index("c")] if "c" in axes else 1
    channel_names = channel_names or [f"ch{i}" for i in range(n_c)]
    if len(channel_names) != n_c:
        raise ValueError(f"expected {n_c} channel names, got {len(channel_names)}")

    root = zarr.group(store=get_store(uri, mode="w"), overwrite=True)
    pyramid = _build_pyramid(data, axes, levels, labels=False)
    chunks = tuple(min(chunk, s) if ax in "yx" else 1 for ax, s in zip(axes, data.shape))
    for level, arr in enumerate(pyramid):
        root.create_dataset(str(level), data=arr, chunks=chunks, compressor=_COMPRESSOR)
    root.attrs["multiscales"] = [_multiscales(axes, len(pyramid), pixel_size_um, name)]
    root.attrs["omero"] = _omero(data, axes, channel_names)
    if extra_attrs:
        root.attrs["phenoscope"] = extra_attrs
    return root


def write_labels(labels: np.ndarray, uri: str, name: str = "cells", chunk: int = 512) -> None:
    """Attach a label image (``yx``) to an existing OME-Zarr image under ``labels/<name>``."""
    root = zarr.open_group(store=get_store(uri, mode="a"), mode="a")
    image_ms = root.attrs["multiscales"][0]
    image_axes = "".join(a["name"] for a in image_ms["axes"])
    spatial = [
        ds_scale
        for ax, ds_scale in zip(image_axes, image_ms["datasets"][0]["coordinateTransformations"][0]["scale"])
        if ax in "zyx"
    ]
    label_axes = "yx" if labels.ndim == 2 else "zyx"
    group = root.require_group("labels")
    existing = list(group.attrs.get("labels", []))
    if name not in existing:
        existing.append(name)
    group.attrs["labels"] = existing

    lg = group.create_group(name, overwrite=True)
    pyramid = _build_pyramid(labels.astype(np.uint32), label_axes, 4, labels=True)
    chunks = tuple(min(chunk, s) for s in labels.shape)
    for level, arr in enumerate(pyramid):
        lg.create_dataset(str(level), data=arr, chunks=chunks, compressor=_COMPRESSOR)
    lg.attrs["multiscales"] = [
        _multiscales(label_axes, len(pyramid), tuple(spatial[-labels.ndim:]), name)
    ]
    lg.attrs["image-label"] = {"version": NGFF_VERSION, "source": {"image": "../../"}}


def read_ome_zarr(uri: str, level: int = 0) -> OmeZarrImage:
    root = zarr.open_group(store=get_store(uri), mode="r")
    ms = root.attrs["multiscales"][0]
    axes = "".join(a["name"] for a in ms["axes"])
    ds = ms["datasets"][level]
    scale = ds["coordinateTransformations"][0]["scale"]
    pixel_size = tuple(float(s) for ax, s in zip(axes, scale) if ax in "zyx")
    channels = [c["label"] for c in root.attrs.get("omero", {}).get("channels", [])]
    data = root[ds["path"]][:]
    return OmeZarrImage(
        data=data,
        axes=axes,
        channel_names=channels or [f"ch{i}" for i in range(data.shape[axes.index("c")])],
        pixel_size_um=pixel_size,
        level=level,
        attrs=dict(root.attrs.get("phenoscope", {})),
    )


def read_labels(uri: str, name: str = "cells", level: int = 0) -> np.ndarray:
    root = zarr.open_group(store=get_store(uri), mode="r")
    return root["labels"][name][str(level)][:]


def list_levels(uri: str) -> int:
    root = zarr.open_group(store=get_store(uri), mode="r")
    return len(root.attrs["multiscales"][0]["datasets"])
