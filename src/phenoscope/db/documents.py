"""Schemaless acquisition metadata in MongoDB (plate maps, microscope settings, OMERO dumps)."""

from __future__ import annotations

from typing import Any

from phenoscope.config import get_settings


class MetadataStore:
    def __init__(self, url: str | None = None, db: str | None = None) -> None:
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("MongoDB support needs `pip install phenoscope[mongo]`") from exc
        settings = get_settings()
        self._db = MongoClient(url or settings.mongo_url)[db or settings.mongo_db]
        self._db.acquisitions.create_index([("image_uri", 1)], unique=True)
        self._db.plate_maps.create_index([("plate", 1), ("well", 1)], unique=True)

    def upsert_acquisition(self, image_uri: str, meta: dict[str, Any]) -> None:
        self._db.acquisitions.update_one(
            {"image_uri": image_uri}, {"$set": {"image_uri": image_uri, **meta}}, upsert=True
        )

    def get_acquisition(self, image_uri: str) -> dict[str, Any] | None:
        return self._db.acquisitions.find_one({"image_uri": image_uri}, {"_id": 0})

    def upsert_plate_map(self, plate: str, rows: list[dict[str, Any]]) -> int:
        for row in rows:
            self._db.plate_maps.update_one(
                {"plate": plate, "well": row["well"]}, {"$set": {"plate": plate, **row}}, upsert=True
            )
        return len(rows)

    def plate_map(self, plate: str) -> list[dict[str, Any]]:
        return list(self._db.plate_maps.find({"plate": plate}, {"_id": 0}))
