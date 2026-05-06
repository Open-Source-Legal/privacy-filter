from __future__ import annotations

from collections.abc import Callable

from .protocol import Detection

DetectionScript = list[Detection] | Callable[[str], list[Detection]]


class FakeDetector:
    """In-memory detector for tests. Returns either a fixed list or runs a callable."""

    def __init__(
        self,
        script: DetectionScript,
        *,
        model_id: str = "fake",
        model_revision: str = "test",
    ) -> None:
        self._script = script
        self._model_id = model_id
        self._model_revision = model_revision

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._model_revision

    def detect(self, text: str) -> list[Detection]:
        if callable(self._script):
            return self._script(text)
        return list(self._script)
