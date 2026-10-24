"""Storage abstraction: the same code path handles local paths and ``s3://`` URIs."""

from __future__ import annotations

from collections.abc import MutableMapping

import fsspec

from phenoscope.config import get_settings


def join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    return "/".join([base, *(p.strip("/") for p in parts)])


def is_remote(uri: str) -> bool:
    return "://" in uri and not uri.startswith("file://")


def get_store(uri: str, mode: str = "r") -> MutableMapping:
    """Return a zarr-compatible key/value store for ``uri``.

    Remote stores use ``s3fs`` (``pip install phenoscope[cloud]``); an optional custom
    endpoint (MinIO, Ceph) is taken from ``PHENOSCOPE_S3_ENDPOINT_URL``.
    """
    storage_options: dict = {}
    if uri.startswith("s3://"):
        settings = get_settings()
        client_kwargs = {"region_name": settings.s3_region}
        if settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = settings.s3_endpoint_url
        storage_options["client_kwargs"] = client_kwargs
    mapper = fsspec.get_mapper(uri, **storage_options)
    if mode != "r" and not is_remote(uri):
        mapper.fs.makedirs(mapper.root, exist_ok=True)
    return mapper
