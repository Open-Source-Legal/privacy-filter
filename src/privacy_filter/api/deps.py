from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from privacy_filter.config import Settings
from privacy_filter.detection.protocol import Detector
from privacy_filter.security import is_valid_api_key

from .errors import InvalidAPIKey


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_detector(request: Request) -> Detector:
    detector: Detector = request.app.state.detector
    return detector


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if not is_valid_api_key(x_api_key, settings.api_keys):
        raise InvalidAPIKey()
