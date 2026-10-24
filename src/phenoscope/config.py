"""Runtime configuration, loaded from environment variables (``PHENOSCOPE_*``) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PHENOSCOPE_", env_file=".env", extra="ignore")

    # Storage
    data_root: str = Field("./data", description="Local dir or s3:// URI for OME-Zarr stores.")
    s3_endpoint_url: str | None = Field(None, description="Custom endpoint (e.g. MinIO).")
    s3_region: str = "eu-central-1"

    # Metadata stores
    database_url: str = "sqlite:///./phenoscope.db"
    mongo_url: str | None = None
    mongo_db: str = "phenoscope"

    # Compute
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    job_backend: Literal["local", "kubernetes"] = "local"
    k8s_namespace: str = "phenoscope"
    k8s_gpu_image: str = "ghcr.io/maxxrichard/phenoscope-gpu:latest"
    max_local_workers: int = 2

    # Models
    model_dir: str = "./models"
    default_segmenter: str = "watershed"
    vit_model: str = "vit_small_patch16_224.dino"

    # Imaging defaults
    tile_size: int = 1024
    zarr_chunk: int = 512
    pyramid_levels: int = 4

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device string; falls back to CPU when torch is unavailable."""
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
