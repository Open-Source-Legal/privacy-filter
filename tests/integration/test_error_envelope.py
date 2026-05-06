import pytest
from httpx import ASGITransport, AsyncClient

from privacy_filter.api.app import create_app
from privacy_filter.config import Settings
from privacy_filter.detection.fake import FakeDetector
from privacy_filter.detection.protocol import Detection


class _BoomDetector(FakeDetector):
    """Succeeds during lifespan warmup but raises on every other call.

    The app's lifespan invokes ``detector.detect("warmup")`` before yielding;
    if that raised, we'd never reach the real /v1/detect call. The guard lets
    warmup pass and routes a real request into the catch-all exception handler
    so we can verify the error envelope.
    """

    def detect(self, text: str) -> list[Detection]:
        if text == "warmup":
            return []
        raise RuntimeError("synthetic failure: alice@example.com")


@pytest.mark.asyncio
async def test_unhandled_error_returns_generic_envelope(settings: Settings, api_key: str) -> None:
    def factory(_s: Settings) -> _BoomDetector:
        return _BoomDetector([], model_id="fake", model_revision="r1")

    app = create_app(settings=settings, detector_factory=factory)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c,
    ):
        r = await c.post(
            "/v1/detect",
            headers={"X-API-Key": api_key},
            json={"text": "hi"},
        )
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["code"] == "internal_error"
        assert body["error"]["message"] == "Internal server error."
        text = r.text
        assert "synthetic failure" not in text
        assert "Traceback" not in text
        assert "RuntimeError" not in text
