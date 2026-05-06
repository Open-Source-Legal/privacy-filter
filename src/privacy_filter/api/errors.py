from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    UNAUTHORIZED = "unauthorized"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    INVALID_REQUEST = "invalid_request"
    NOT_READY = "not_ready"
    INTERNAL_ERROR = "internal_error"


class APIError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500
    message: str = "Internal server error."


class InvalidAPIKey(APIError):
    code = ErrorCode.UNAUTHORIZED
    status_code = 401
    message = "Invalid or missing API key."


class PayloadTooLarge(APIError):
    code = ErrorCode.PAYLOAD_TOO_LARGE
    status_code = 413
    message = "Request body exceeds the maximum allowed size."


class InvalidRequest(APIError):
    code = ErrorCode.INVALID_REQUEST
    status_code = 422
    message = "Request body is invalid."


class ServiceNotReady(APIError):
    code = ErrorCode.NOT_READY
    status_code = 503
    message = "Service is not ready."


def error_envelope(exc: Exception, *, request_id: str) -> dict[str, Any]:
    if isinstance(exc, APIError):
        code = exc.code.value
        message = exc.message
    else:
        code = ErrorCode.INTERNAL_ERROR.value
        message = "Internal server error."
    return {"error": {"code": code, "message": message, "request_id": request_id}}
