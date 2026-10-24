"""FastAPI backend serving bench scientists, bioimage analysts and the web/napari viewers."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from phenoscope import __version__
from phenoscope.api.routes import datasets, drift, images, jobs, system
from phenoscope.db.session import init_db

STATIC = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="PhenoScope API",
        version=__version__,
        description="Segmentation, phenotypic profiling, registration and drift monitoring "
        "for microscopy and digital pathology.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    for r in (system.router, datasets.router, images.router, jobs.router, drift.router):
        app.include_router(r)

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app


app = create_app()
