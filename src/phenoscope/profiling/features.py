"""Hand-crafted per-object features (morphology, intensity, texture).

These complement the learned ViT embeddings: they are interpretable, cheap, and
the reference point the drift monitor uses for staining/intensity shifts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, regionprops_table

MORPHOLOGY = (
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "centroid",
    "bbox",
)
INTENSITY = ("intensity_mean", "intensity_max", "intensity_min", "intensity_std")


def _intensity_std(mask: np.ndarray, intensity: np.ndarray) -> float:
    return float(intensity[mask].std())


def _texture(labels: np.ndarray, channel: np.ndarray, levels: int = 32) -> pd.DataFrame:
    """Haralick-style GLCM contrast/homogeneity/correlation per object."""
    q = np.digitize(channel, np.quantile(channel, np.linspace(0, 1, levels + 1)[1:-1]))
    rows = []
    for r in regionprops(labels):
        sl = r.slice
        patch = np.where(r.image, q[sl], 0).astype(np.uint8)
        if min(patch.shape) < 3:
            rows.append({"label": r.label})
            continue
        glcm = graycomatrix(patch, [1], [0, np.pi / 2], levels=levels, symmetric=True, normed=True)
        rows.append(
            {
                "label": r.label,
                **{f"texture_{p}": float(graycoprops(glcm, p).mean())
                   for p in ("contrast", "homogeneity", "correlation", "energy")},
            }
        )
    return pd.DataFrame(rows)


def extract_features(
    image: np.ndarray,
    labels: np.ndarray,
    channel_names: Sequence[str] | None = None,
    pixel_size_um: float = 1.0,
    texture: bool = False,
) -> pd.DataFrame:
    """Per-object features for a ``(c, y, x)`` image and a label image.

    Columns are prefixed by channel name, e.g. ``DAPI_intensity_mean``.
    """
    if image.ndim == 2:
        image = image[None]
    channel_names = list(channel_names or [f"ch{i}" for i in range(image.shape[0])])

    df = pd.DataFrame(regionprops_table(labels, properties=MORPHOLOGY))
    if df.empty:
        return df
    for col in ("area",):
        df[f"{col}_um2"] = df[col] * pixel_size_um**2
    df["perimeter_um"] = df["perimeter"] * pixel_size_um
    df["form_factor"] = 4 * np.pi * df["area"] / np.maximum(df["perimeter"] ** 2, 1e-6)

    for c, name in enumerate(channel_names):
        t = pd.DataFrame(
            regionprops_table(
                labels,
                intensity_image=image[c],
                properties=("label", *INTENSITY[:3]),
                extra_properties=(_intensity_std,),
            )
        ).rename(columns={"_intensity_std": "intensity_std"})
        t["intensity_integrated"] = t["intensity_mean"] * df["area"].to_numpy()
        t = t.rename(columns={k: f"{name}_{k}" for k in t.columns if k != "label"})
        df = df.merge(t, on="label")
        if texture:
            tex = _texture(labels, image[c])
            tex = tex.rename(columns={k: f"{name}_{k}" for k in tex.columns if k != "label"})
            df = df.merge(tex, on="label", how="left")
    return df
