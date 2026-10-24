from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from phenoscope import __version__
from phenoscope.config import get_settings
from phenoscope.db.session import get_session
from phenoscope.segmentation import available_segmenters

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    db_ok = True
    try:
        with get_session() as s:
            s.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "version": __version__, "database": db_ok}


@router.get("/segmenters")
def segmenters() -> dict:
    return {"available": available_segmenters(), "default": get_settings().default_segmenter}
