from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from privacy_filter.api.app import create_app
from privacy_filter.config import Settings
from privacy_filter.detection.fake import FakeDetector
from privacy_filter.detection.protocol import Detection, Label


@pytest.fixture
def api_key() -> str:
    return "test-key"


@pytest.fixture
def settings(api_key: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("API_KEYS", api_key)
    monkeypatch.setenv("MAX_INPUT_CHARS", "100")
    monkeypatch.setenv("MAX_BODY_BYTES", "4096")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    return Settings()


@pytest.fixture
def fake_detector_factory() -> Callable[[Settings], FakeDetector]:
    def factory(_settings: Settings) -> FakeDetector:
        def script(text: str) -> list[Detection]:
            target = "alice@example.com"
            if target in text:
                start = text.index(target)
                end = start + len(target)
                return [
                    Detection(
                        entity_group=Label.PRIVATE_EMAIL,
                        start=start,
                        end=end,
                        score=0.99,
                        word=text[start:end],
                    )
                ]
            return []

        return FakeDetector(script, model_id="fake", model_revision="rev-test")

    return factory


@pytest.fixture
def app(
    settings: Settings,
    fake_detector_factory: Callable[[Settings], FakeDetector],
) -> FastAPI:
    return create_app(settings=settings, detector_factory=fake_detector_factory)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c,
    ):
        yield c
