"""ASGI entry point.

Run with::

    uvicorn api.main:app --reload --port 8000

The service starts immediately and loads the graph on a background thread, so
``/api/health`` is answerable within milliseconds and reports ``status: "warming"``
until the model is scored. With no dataset present it starts directly in artifact
mode and serves the committed JSON.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.state import get_backend
from fraudlens import __version__

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraudlens.api")

# The Vite dev server and common static hosts. Credentials are never used, so a
# permissive origin list costs nothing here.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://localhost:3000",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = get_backend()
    log.info("Starting in %s mode", backend.mode)
    backend.start_loading()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="FraudLens API",
        version=__version__,
        description=(
            "Graph neural network fraud detection on the Elliptic Bitcoin dataset. "
            "Serves live model inference when the dataset is present, and precomputed "
            "artifacts otherwise."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=r"https?://localhost(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "fraudlens",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
