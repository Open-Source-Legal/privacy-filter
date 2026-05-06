"""Placeholder stub for the real HuggingFace token-classification detector.

Task 17 replaces this module with the real torch/transformers implementation.
Kept here so the ``Detector`` Protocol type is satisfied at import/typecheck
time without requiring heavy ML dependencies in test runs.
"""

from __future__ import annotations

from .protocol import Detection


class HuggingFaceDetector:
    """Placeholder for the real HF token-classification detector.

    Construction raises ``NotImplementedError`` so that any code path that
    actually instantiates this class (rather than just importing the symbol)
    fails loudly until Task 17 lands.
    """

    def __init__(self, *, model_id: str, revision: str | None = None) -> None:
        raise NotImplementedError("HuggingFaceDetector is not yet implemented (see Task 17).")

    @property
    def model_id(self) -> str:  # pragma: no cover - placeholder
        raise NotImplementedError

    @property
    def model_revision(self) -> str:  # pragma: no cover - placeholder
        raise NotImplementedError

    def detect(self, text: str) -> list[Detection]:  # pragma: no cover - placeholder
        raise NotImplementedError
