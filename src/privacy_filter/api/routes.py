from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from pydantic import ValidationError as _PydanticValidationError

from privacy_filter.config import Settings
from privacy_filter.detection.protocol import Detector

from .deps import get_detector, get_settings, require_api_key
from .errors import InvalidRequest, ServiceNotReady
from .schemas import DetectionOut, DetectRequest, DetectResponse, bounded_detect_request

log = structlog.get_logger(__name__)

router = APIRouter()
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(
    settings: Annotated[Settings, Depends(get_settings)],  # forces lifespan
    detector: Annotated[Detector | None, Depends(get_detector)] = None,
) -> dict[str, str]:
    if detector is None:
        raise ServiceNotReady()
    return {"status": "ready"}


@v1.post("/detect", response_model=DetectResponse)
def detect(
    body: DetectRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    detector: Annotated[Detector, Depends(get_detector)],
) -> DetectResponse:
    bounded = bounded_detect_request(max_chars=settings.max_input_chars)
    try:
        body = bounded.model_validate(body.model_dump())
    except _PydanticValidationError as e:
        raise InvalidRequest() from e

    detections = detector.detect(body.text)

    log.info(
        "detect_completed",
        endpoint="/v1/detect",
        method="POST",
        status=200,
        input_chars=len(body.text),
        detection_count=len(detections),
    )

    return DetectResponse(
        detections=[
            DetectionOut(
                entity_group=d.entity_group.value,
                score=d.score,
                word=d.word,
                start=d.start,
                end=d.end,
            )
            for d in detections
        ],
        model=detector.model_id,
        model_revision=detector.model_revision,
    )
