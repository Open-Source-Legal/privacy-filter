"""HuggingFace token-classification detector for openai/privacy-filter.

Imports of ``transformers`` and ``torch`` are deferred to construction so
the rest of the codebase can be type-checked and tested without the
optional ``[hf]`` extra installed.
"""

from __future__ import annotations

from typing import Any

from .bioes import TaggedToken, group_bioes
from .protocol import Detection


class HuggingFaceDetector:
    def __init__(self, *, model_id: str, revision: str | None = None) -> None:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            device_map="auto",
            **kwargs,
        )
        self._pipeline = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="none",
        )
        self._model_id = model_id
        resolved = getattr(model.config, "_commit_hash", None) or revision or "unknown"
        self._revision = str(resolved)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._revision

    def detect(self, text: str) -> list[Detection]:
        if not text:
            return []
        raw = self._pipeline(text)
        tokens = [
            TaggedToken(
                tag=str(item["entity"]),
                score=float(item["score"]),
                start=int(item["start"]),
                end=int(item["end"]),
            )
            for item in raw
        ]
        return group_bioes(text, tokens)
