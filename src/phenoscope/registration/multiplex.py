"""Cyclic multiplexed-imaging registration (CyCIF / CODEX-style, e.g. 12 markers over 4 cycles).

Every cycle re-images a nuclear reference stain (DAPI/Hoechst). Each cycle is
aligned to cycle 0 on that channel in two stages:

1. Coarse translation via phase cross-correlation on a downsampled image.
2. Optional affine refinement with OpenCV's ECC maximisation.

The resulting transform is applied to every marker channel of the cycle and the
markers are stacked into one registered ``(c, y, x)`` image.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from phenoscope.preprocessing.normalize import percentile_normalize


@dataclass
class CycleImage:
    data: np.ndarray  # (c, y, x)
    channel_names: list[str]
    reference_channel: str = "DAPI"

    @property
    def reference(self) -> np.ndarray:
        return self.data[self.channel_names.index(self.reference_channel)]


@dataclass
class RegistrationResult:
    image: np.ndarray
    channel_names: list[str]
    transforms: list[np.ndarray] = field(default_factory=list)  # 2x3 affine per cycle
    ncc_before: list[float] = field(default_factory=list)
    ncc_after: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return {
            "n_markers": len(self.channel_names),
            "n_cycles": len(self.transforms),
            "mean_ncc_before": float(np.mean(self.ncc_before)),
            "mean_ncc_after": float(np.mean(self.ncc_after)),
        }


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def register_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    downsample: int = 4,
    affine: bool = True,
    ecc_iters: int = 200,
) -> np.ndarray:
    """Return a 2x3 affine aligning ``moving`` to ``fixed``.

    The matrix uses OpenCV's ECC convention (fixed → moving coordinates), i.e. it is
    applied with ``WARP_INVERSE_MAP`` (see :func:`apply_transform`).
    """
    f = percentile_normalize(fixed).astype(np.float32)
    m = percentile_normalize(moving).astype(np.float32)
    shift, _, _ = phase_cross_correlation(
        f[::downsample, ::downsample], m[::downsample, ::downsample], upsample_factor=10
    )
    dy, dx = shift * downsample
    warp = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)

    # Sub-pixel translation refinement at full resolution around the coarse estimate.
    fine, _, _ = phase_cross_correlation(f, apply_transform(m, warp), upsample_factor=20)
    warp[0, 2] -= fine[1]
    warp[1, 2] -= fine[0]

    if affine:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iters, 1e-6)
        # If ECC fails to converge, keep the translation estimate.
        with contextlib.suppress(cv2.error):
            _, warp = cv2.findTransformECC(
                cv2.GaussianBlur(f, (5, 5), 0),
                cv2.GaussianBlur(m, (5, 5), 0),
                warp.copy(),
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )
    return warp


def apply_transform(img: np.ndarray, warp: np.ndarray) -> np.ndarray:
    h, w = img.shape[-2:]
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    if img.ndim == 2:
        return cv2.warpAffine(img.astype(np.float32), warp, (w, h), flags=flags).astype(img.dtype)
    return np.stack([apply_transform(c, warp) for c in img])


def register_cycles(
    cycles: Sequence[CycleImage],
    keep_reference: str = "first",
    affine: bool = True,
) -> RegistrationResult:
    """Register all cycles to ``cycles[0]`` and stack marker channels.

    ``keep_reference``: ``"first"`` keeps only cycle 0's nuclear channel,
    ``"all"`` keeps each cycle's reference (useful for QC).
    """
    fixed = cycles[0].reference
    channels: list[np.ndarray] = []
    names: list[str] = []
    transforms, before, after = [], [], []

    for i, cyc in enumerate(cycles):
        if i == 0:
            warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        else:
            warp = register_pair(fixed, cyc.reference, affine=affine)
        aligned = apply_transform(cyc.data, warp)
        transforms.append(warp)
        before.append(ncc(fixed, cyc.reference))
        after.append(ncc(fixed, aligned[cyc.channel_names.index(cyc.reference_channel)]))

        for name, plane in zip(cyc.channel_names, aligned):
            is_ref = name == cyc.reference_channel
            if is_ref and not (i == 0 or keep_reference == "all"):
                continue
            channels.append(plane)
            names.append(f"{name}_c{i}" if is_ref and keep_reference == "all" else name)

    return RegistrationResult(
        image=np.stack(channels),
        channel_names=names,
        transforms=transforms,
        ncc_before=before,
        ncc_after=after,
    )
