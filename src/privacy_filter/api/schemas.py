from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Effectively-unbounded default upper bound; per-request size enforced via the
# bounded_detect_request factory in routes.py using the configured Settings.
_DEFAULT_MAX_CHARS = 10**9


class DetectRequest(BaseModel):
    """Request body for POST /v1/detect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: Annotated[
        str,
        StringConstraints(min_length=1, max_length=_DEFAULT_MAX_CHARS),
    ]


def bounded_detect_request(*, max_chars: int) -> type[DetectRequest]:
    """Build a DetectRequest subclass with a tightened ``text`` length bound.

    Used by the /v1/detect route to enforce ``Settings.max_input_chars`` on
    incoming requests without mutating the base model.
    """

    class _Bounded(DetectRequest):
        text: Annotated[
            str,
            StringConstraints(min_length=1, max_length=max_chars),
        ]

    _Bounded.__name__ = "DetectRequest"
    return _Bounded


class DetectionOut(BaseModel):
    """One PII detection on the wire. Mirrors HF token-classification
    pipeline grouped output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_group: str
    score: float = Field(ge=0.0, le=1.0)
    word: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class DetectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detections: list[DetectionOut]
    model: str
    model_revision: str
