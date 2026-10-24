"""Relational metadata (PostgreSQL in production, SQLite for local/dev).

Pixel data never lives here, only URIs to OME-Zarr stores. Free-form acquisition
metadata (microscope settings, plate maps, OMERO exports) goes to MongoDB via
:mod:`phenoscope.db.documents`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    modality: Mapped[str] = mapped_column(String(32))  # fluorescence|brightfield|he|multiplex
    description: Mapped[str] = mapped_column(Text, default="")
    uri: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    images: Mapped[list[Image]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    uri: Mapped[str] = mapped_column(String(1024))
    plate: Mapped[str | None] = mapped_column(String(64), index=True)
    well: Mapped[str | None] = mapped_column(String(8))
    field_of_view: Mapped[int | None] = mapped_column(Integer)
    sample_id: Mapped[str | None] = mapped_column(String(128), index=True)
    batch_id: Mapped[str | None] = mapped_column(String(128), index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    channels: Mapped[dict[str, Any]] = mapped_column(default=dict)
    pixel_size_um: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    dataset: Mapped[Dataset] = relationship(back_populates="images")
    segmentations: Mapped[list[SegmentationRun]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class SegmentationRun(Base):
    __tablename__ = "segmentation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), index=True)
    model: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64), default="")
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    label_name: Mapped[str] = mapped_column(String(64))
    n_objects: Mapped[int] = mapped_column(Integer)
    runtime_s: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    image: Mapped[Image] = relationship(back_populates="segmentations")


class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id: Mapped[int] = mapped_column(primary_key=True)
    segmentation_id: Mapped[int] = mapped_column(ForeignKey("segmentation_runs.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # handcrafted | vit
    uri: Mapped[str] = mapped_column(String(1024))  # parquet / zarr path
    n_objects: Mapped[int] = mapped_column(Integer)
    n_features: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DriftCheck(Base):
    __tablename__ = "drift_checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark: Mapped[str] = mapped_column(String(128), index=True)
    batch_id: Mapped[str] = mapped_column(String(128), index=True)
    drifted: Mapped[bool]
    mmd2: Mapped[float | None] = mapped_column(Float)
    mmd_p: Mapped[float | None] = mapped_column(Float)
    report: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    backend: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
