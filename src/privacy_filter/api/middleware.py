from __future__ import annotations

import uuid

import structlog
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import PayloadTooLarge, error_envelope


def _get_header(scope: Scope, name: bytes) -> bytes | None:
    """Return the first matching header value from an ASGI scope, or None."""
    headers: list[tuple[bytes, bytes]] = scope.get("headers") or []
    for k, v in headers:
        if k == name:
            return v
    return None


class RequestIDMiddleware:
    """Assigns a request_id (from header or generated) and binds it into the
    structlog context for the duration of the request.

    Implemented as pure ASGI (rather than ``BaseHTTPMiddleware``) so that the
    exception-propagation contract on the wrapped app remains clean.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid_bytes = _get_header(scope, b"x-request-id")
        request_id = rid_bytes.decode("latin-1") if rid_bytes else uuid.uuid4().hex

        # Starlette wires ``request.state`` to ``scope["state"]``; setting the
        # key here lets downstream code read it via ``request.state.request_id``.
        scope_state = scope.setdefault("state", {})
        scope_state["request_id"] = request_id

        structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers") or [])
                response_headers.append(
                    (b"x-request-id", request_id.encode("latin-1")),
                )
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            structlog.contextvars.clear_contextvars()


class SecurityHeadersMiddleware:
    """Appends conservative security response headers (only when not already
    present) without mutating downstream behaviour."""

    _HEADERS: tuple[tuple[bytes, bytes], ...] = (
        (b"x-content-type-options", b"nosniff"),
        (b"referrer-policy", b"no-referrer"),
        (b"x-frame-options", b"DENY"),
    )

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers") or [])
                existing = {k.lower() for k, _ in response_headers}
                for name, value in self._HEADERS:
                    if name not in existing:
                        response_headers.append((name, value))
                message = {**message, "headers": response_headers}
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


class BodySizeLimitMiddleware:
    """Rejects requests whose Content-Length exceeds ``max_bytes`` with a 413
    error envelope before the body is read.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        cl_bytes = _get_header(scope, b"content-length")
        if cl_bytes is not None:
            try:
                if int(cl_bytes) > self._max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                await self._reject(scope, receive, send)
                return

        await self.app(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        # ``RequestIDMiddleware`` runs outermost, so ``scope["state"]`` already
        # carries a request_id by the time we reject here.
        request_id = ""
        state = scope.get("state")
        if isinstance(state, dict):
            request_id = str(state.get("request_id", ""))
        err = PayloadTooLarge()
        response = JSONResponse(
            status_code=err.status_code,
            content=error_envelope(err, request_id=request_id),
        )
        await response(scope, receive, send)


class SuppressHandledExceptionMiddleware:
    """Outermost wrapper that swallows exceptions which Starlette has already
    converted into a sent response.

    Starlette's ``ServerErrorMiddleware`` invokes the registered 500 handler,
    sends the JSON 500 envelope, and **then re-raises** the original exception
    so that ASGI servers can log it and test clients can opt-in to observing
    it. Under ``httpx.ASGITransport`` (which defaults to
    ``raise_app_exceptions=True``) that re-raised exception bubbles to the
    test client and prevents asserting on the response body.

    By wrapping the FastAPI app with a layer outside ``ServerErrorMiddleware``,
    we can detect that a response has already been started and silently absorb
    the re-raised exception. The structured-log error record produced by our
    own ``Exception`` handler still fires, so observability is unaffected.

    If the response was *not* started by the time the exception reaches us
    (which means no handler produced a response — e.g. an exception inside
    ``ServerErrorMiddleware`` itself), we re-raise so the failure is not
    silenced.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            if not response_started:
                raise
