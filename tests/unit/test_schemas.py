import pytest
from pydantic import ValidationError

from privacy_filter.api.schemas import (
    DetectionOut,
    DetectRequest,
    DetectResponse,
    bounded_detect_request,
)


def test_detect_request_accepts_non_empty_text() -> None:
    req = DetectRequest.model_validate({"text": "hello"})
    assert req.text == "hello"


def test_detect_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        DetectRequest.model_validate({"text": ""})


def test_detect_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DetectRequest.model_validate({"text": "hi", "extra": 1})


def test_bounded_detect_request_rejects_text_above_max_chars() -> None:
    Bounded = bounded_detect_request(max_chars=5)
    with pytest.raises(ValidationError):
        Bounded.model_validate({"text": "abcdef"})


def test_bounded_detect_request_accepts_text_at_max_chars() -> None:
    Bounded = bounded_detect_request(max_chars=5)
    obj = Bounded.model_validate({"text": "abcde"})
    assert obj.text == "abcde"


def test_bounded_detect_request_rejects_extra_fields() -> None:
    Bounded = bounded_detect_request(max_chars=10)
    with pytest.raises(ValidationError):
        Bounded.model_validate({"text": "hi", "mode": "redact"})


def test_detection_out_shape() -> None:
    out = DetectionOut(
        entity_group="private_email",
        score=0.9,
        word="alice@x.com",
        start=0,
        end=11,
    )
    assert out.model_dump() == {
        "entity_group": "private_email",
        "score": 0.9,
        "word": "alice@x.com",
        "start": 0,
        "end": 11,
    }


def test_detect_response_shape() -> None:
    resp = DetectResponse(
        detections=[
            DetectionOut(
                entity_group="secret",
                score=0.5,
                word="x",
                start=0,
                end=1,
            )
        ],
        model="openai/privacy-filter",
        model_revision="abc",
    )
    payload = resp.model_dump()
    assert payload["detections"][0]["entity_group"] == "secret"
    assert payload["detections"][0]["word"] == "x"
    assert payload["model_revision"] == "abc"


def test_detect_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DetectResponse.model_validate(
            {
                "detections": [],
                "model": "m",
                "model_revision": "r",
                "redacted": "should not be here",
            }
        )
