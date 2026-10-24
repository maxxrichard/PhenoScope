"""napari widgets (requires ``pip install phenoscope[viewer]``)."""

from __future__ import annotations

import json
import urllib.request

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container, PushButton, Table

from phenoscope.segmentation import available_segmenters, get_segmenter


@magic_factory(
    segmenter={"choices": available_segmenters()},
    min_size={"min": 0, "max": 10_000},
    call_button="Segment",
)
def segment_widget(
    image: napari.layers.Image,  # noqa: F821
    segmenter: str = "watershed",
    diameter: float = 30.0,
    min_size: int = 30,
) -> napari.types.LabelsData:  # noqa: F821
    data = np.asarray(image.data)
    if data.ndim == 2:
        data = data[None]
    kwargs = {"min_size": min_size}
    if segmenter == "cellpose":
        kwargs["diameter"] = diameter
    return get_segmenter(segmenter, **kwargs).segment(data)


@magic_factory(call_button="Measure")
def features_widget(
    image: napari.layers.Image,  # noqa: F821
    labels: napari.layers.Labels,  # noqa: F821
    texture: bool = False,
) -> None:
    from phenoscope.profiling.features import extract_features

    df = extract_features(np.asarray(image.data), np.asarray(labels.data), texture=texture)
    labels.features = df
    labels.properties = df.to_dict(orient="list")


class ServerBrowser(Container):
    """Pull images and label layers from a running PhenoScope API into napari."""

    def __init__(self, napari_viewer, server: str = "http://localhost:8000") -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.server = server
        self.dataset = ComboBox(label="Dataset")
        self.table = Table()
        self.load = PushButton(text="Open selected")
        self.extend([self.dataset, self.table, self.load])
        self.dataset.changed.connect(self._list_images)
        self.load.changed.connect(self._open)
        self._refresh()

    def _get(self, path: str):
        with urllib.request.urlopen(f"{self.server}{path}") as r:  # noqa: S310 - user-provided URL
            return json.load(r)

    def _refresh(self) -> None:
        self.dataset.choices = [(d["name"], d["id"]) for d in self._get("/datasets")]

    def _list_images(self, dataset_id: int) -> None:
        rows = self._get(f"/datasets/{dataset_id}/images")
        self.table.value = [{"id": r["id"], "name": r["name"], "uri": r["uri"]} for r in rows]

    def _open(self) -> None:
        from phenoscope.io.omezarr import read_ome_zarr

        for row in self.table.selection or []:
            uri = self.table.data[row[0]][2] if isinstance(row, tuple) else row["uri"]
            img = read_ome_zarr(uri)
            self.viewer.add_image(img.data, channel_axis=0, name=img.channel_names,
                                  scale=img.pixel_size_um[-2:])
