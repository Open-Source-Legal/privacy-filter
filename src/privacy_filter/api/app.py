from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from privacy_filter.config import Settings
from privacy_filter.detection.protocol import Detector
from privacy_filter.logging import configure_logging

from .errors import APIError, InvalidRequest, error_envelope
from .middleware import (
    BodySizeLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from .routes import router, v1

log = structlog.get_logger(__name__)

DetectorFactory = Callable[[Settings], Detector]


def _default_detector_factory(settings: Settings) -> Detector:
    from privacy_filter.detection.huggingface import HuggingFaceDetector

    return HuggingFaceDetector(
        model_id=settings.model_id,
        revision=settings.model_revision,
    )


def create_app(
    *,
    settings: Settings | None = None,
    detector_factory: DetectorFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    configure_logging(level=resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        factory = detector_factory or _default_detector_factory
        t0 = time.perf_counter()
        app.state.detector = factory(resolved_settings)
        # Warm-up: a single short inference; failures here block readiness.
        app.state.detector.detect("warmup")
        log.info(
            "lifespan_ready",
            endpoint="lifespan",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        try:
            yield
        finally:
            app.state.detector = None

    app = FastAPI(title="privacy-filter", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=resolved_settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)
    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_methods=["POST", "GET"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    app.include_router(router)
    app.include_router(v1)

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(exc, request_id=rid),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(
        request: Request,
        exc: RequestValidationError,  # FastAPI requires this signature
    ) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        return JSONResponse(
            status_code=InvalidRequest.status_code,
            content=error_envelope(InvalidRequest(), request_id=rid),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        log.error(
            "unhandled_exception",
            exc_class=type(exc).__name__,
            code="internal_error",
        )
        return JSONResponse(status_code=500, content=error_envelope(exc, request_id=rid))

    return app
