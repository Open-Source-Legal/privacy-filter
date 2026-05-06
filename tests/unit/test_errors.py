import pytest

from privacy_filter.api.errors import (
    APIError,
    ErrorCode,
    InvalidAPIKey,
    PayloadTooLarge,
    error_envelope,
)


def test_api_error_subclasses_carry_code_and_status() -> None:
    assert InvalidAPIKey().code is ErrorCode.UNAUTHORIZED
    assert InvalidAPIKey().status_code == 401
    assert PayloadTooLarge().code is ErrorCode.PAYLOAD_TOO_LARGE
    assert PayloadTooLarge().status_code == 413


def test_envelope_shape_for_known_error() -> None:
    err = InvalidAPIKey()
    payload = error_envelope(err, request_id="req-123")
    assert payload == {
        "error": {
            "code": "unauthorized",
            "message": "Invalid or missing API key.",
            "request_id": "req-123",
        }
    }


def test_envelope_for_internal_error_is_generic() -> None:
    payload = error_envelope(RuntimeError("DB exploded"), request_id="req-9")
    assert payload == {
        "error": {
            "code": "internal_error",
            "message": "Internal server error.",
            "request_id": "req-9",
        }
    }
    # Crucial: do not leak the original message.
    assert "DB exploded" not in payload["error"]["message"]


def test_api_error_is_raisable() -> None:
    with pytest.raises(APIError):
        raise InvalidAPIKey()
