from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetCreate(BaseModel):
    name: str
    modality: Literal["fluorescence", "brightfield", "he", "multiplex"]
    uri: str
    description: str = ""


class DatasetOut(DatasetCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    n_images: int = 0


class ImageRegister(BaseModel):
    name: str
    uri: str = Field(..., description="OME-Zarr URI (local path or s3://)")
    plate: str | None = None
    well: str | None = None
    field_of_view: int | None = None
    sample_id: str | None = None
    batch_id: str | None = None


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    dataset_id: int
    name: str
    uri: str
    plate: str | None
    well: str | None
    sample_id: str | None
    batch_id: str | None
    width: int
    height: int
    channels: dict[str, Any]
    pixel_size_um: float


class SegmentRequest(BaseModel):
    segmenter: str = "watershed"
    params: dict[str, Any] = Field(default_factory=dict)
    channels: list[str] | None = None
    label_name: str = "cells"
    asynchronous: bool = False


class SegmentationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    image_id: int
    model: str
    label_name: str
    n_objects: int
    runtime_s: float
    params: dict[str, Any]


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    backend: str
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: str | None = None


class DriftRequest(BaseModel):
    benchmark: str
    batch_id: str
    reference: list[dict[str, float]]
    batch: list[dict[str, float]]
    features: list[str] | None = None


class FeatureDriftOut(BaseModel):
    feature: str
    psi: float
    ks_stat: float
    ks_p: float
    level: str


class DriftOut(BaseModel):
    batch_id: str
    drifted: bool
    reasons: list[str]
    top_features: list[FeatureDriftOut]
