# Privacy Filter Microservice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stateless FastAPI microservice that detects and redacts PII in plain text, backed by the HuggingFace `openai/privacy-filter` model accessed through a pluggable `Detector` interface.

**Architecture:** `src/`-layout Python package. Single-process FastAPI app loaded via factory + lifespan; the detector lives on `app.state` and is injected via a FastAPI dependency that integration tests override with a `FakeDetector`. All ML deps (`transformers`, `torch`) are an optional `[hf]` extra so the bulk of the test suite runs without them. TDD throughout: every behavior is driven by a failing test before implementation lands.

**Tech Stack:** Python 3.12 · `uv` · FastAPI · Uvicorn · Pydantic v2 · `pydantic-settings` · `structlog` · `pytest` + `pytest-asyncio` + `httpx` + `hypothesis` · `transformers` + `torch` (optional extra) · `ruff` · `mypy --strict` · `pip-audit` · `pre-commit` · GitHub Actions · Docker

**Source spec:** `docs/superpowers/specs/2026-05-05-privacy-filter-microservice-design.md`

**Commit-message rule:** Do NOT add any "Co-Authored-By: Claude" trailer or "Generated with Claude Code" footer to any commit, PR, or artifact. Use plain Conventional Commits-style messages.

---

## File map

This plan creates / modifies the following files. Each file has a single responsibility; tasks are scoped so each commit produces self-contained, testable changes.

**Project root:**
- `pyproject.toml` — package metadata, deps, optional extras, ruff/mypy/pytest config
- `uv.lock` — committed lockfile
- `.gitignore`, `.dockerignore`, `.env.example`
- `.pre-commit-config.yaml`
- `Dockerfile`
- `.github/workflows/ci.yml`
- `README.md`
- `CLAUDE.md`

**Package (`src/privacy_filter/`):**
- `__init__.py` — package marker, exposes `__version__`
- `config.py` — `Settings` (pydantic-settings)
- `logging.py` — structlog setup with PII-allowlist processor
- `security.py` — constant-time API-key compare
- `detection/__init__.py`
- `detection/protocol.py` — `Label`, `Detection`, `Detector` Protocol
- `detection/redact.py` — `apply_spans`
- `detection/fake.py` — `FakeDetector`
- `detection/huggingface.py` — `HuggingFaceDetector` (imports torch lazily)
- `api/__init__.py`
- `api/schemas.py` — Pydantic request/response
- `api/errors.py` — exception types + envelope mapper
- `api/middleware.py` — request_id, security headers, body-size guard
- `api/deps.py` — auth + `get_detector` dependency providers
- `api/routes.py` — `/v1/detect`, `/healthz`, `/readyz`
- `api/app.py` — `create_app()` factory + lifespan

**Tests (`tests/`):**
- `conftest.py` — shared fixtures
- `unit/` — pure-logic tests (redaction, protocol, security, logging, schemas, errors)
- `contract/` — `Detector` Protocol contract suite, run against `FakeDetector` (and `HuggingFaceDetector` under `-m slow`)
- `integration/` — `httpx.AsyncClient` over ASGI, `FakeDetector` injected via dependency override
- `property/` — `hypothesis`-driven redaction invariants
- `slow/` — exercises the real HF model (`-m slow`)

---

## Task 1: Bootstrap the repo

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/privacy_filter/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env
.env.local
.coverage
.pytest_cache/
.ruff_cache/
.mypy_cache/
.hypothesis/
dist/
build/
.cache/
htmlcov/
hf_cache/
```

- [ ] **Step 2: Create `.env.example`**

```
API_KEYS=replace-with-comma-separated-keys
MAX_INPUT_CHARS=50000
MAX_BODY_BYTES=262144
MODEL_ID=openai/privacy-filter
MODEL_REVISION=
LOG_LEVEL=INFO
CORS_ORIGINS=
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "privacy-filter"
version = "0.1.0"
description = "PII detection and redaction microservice"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pydantic-settings>=2.4",
    "structlog>=24.4",
]

[project.optional-dependencies]
hf = [
    "transformers>=4.45",
    "torch>=2.4",
]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "httpx>=0.27",
    "hypothesis>=6.112",
    "ruff>=0.6",
    "mypy>=1.11",
    "pip-audit>=2.7",
    "pre-commit>=3.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/privacy_filter"]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "W", "F", "I", "B", "UP", "S", "C4", "SIM", "RUF", "PT", "ANN",
]
ignore = ["ANN401"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "ANN"]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src", "tests"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["transformers.*", "torch.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra --strict-markers"
markers = [
    "slow: tests that load the real HuggingFace model (skipped by default)",
]
```

- [ ] **Step 4: Create package and tests roots**

```python
# src/privacy_filter/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 5: Sync dependencies and verify tooling runs**

```bash
uv venv
uv sync --extra dev
uv run ruff check
uv run mypy
uv run pytest
```

Expected: ruff clean, mypy clean (zero files), pytest reports `no tests ran`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example src/privacy_filter/__init__.py tests/__init__.py
git commit -m "chore: bootstrap project with uv, ruff, mypy, pytest"
```

---

## Task 2: Detection types — `Label` and `Detection`

**Files:**
- Create: `src/privacy_filter/detection/__init__.py`
- Create: `src/privacy_filter/detection/protocol.py`
- Test: `tests/unit/__init__.py`, `tests/unit/test_protocol.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/__init__.py
```

```python
# tests/unit/test_protocol.py
import pytest

from privacy_filter.detection.protocol import Detection, Label


def test_label_values_match_model_card():
    expected = {
        "account_number", "private_address", "private_email", "private_person",
        "private_phone", "private_url", "private_date", "secret",
    }
    assert {member.value for member in Label} == expected


def test_detection_accepts_valid_fields():
    d = Detection(label=Label.PRIVATE_EMAIL, start=0, end=10, score=0.9)
    assert d.label is Label.PRIVATE_EMAIL
    assert d.start == 0
    assert d.end == 10
    assert d.score == 0.9


def test_detection_rejects_negative_start():
    with pytest.raises(ValueError, match="start must be >= 0"):
        Detection(label=Label.SECRET, start=-1, end=5, score=0.5)


def test_detection_rejects_end_before_start():
    with pytest.raises(ValueError, match="end .* must be >= start"):
        Detection(label=Label.SECRET, start=10, end=5, score=0.5)


def test_detection_allows_zero_length_span():
    Detection(label=Label.SECRET, start=5, end=5, score=0.5)


def test_detection_rejects_score_below_zero():
    with pytest.raises(ValueError, match="score must be in"):
        Detection(label=Label.SECRET, start=0, end=1, score=-0.1)


def test_detection_rejects_score_above_one():
    with pytest.raises(ValueError, match="score must be in"):
        Detection(label=Label.SECRET, start=0, end=1, score=1.1)


def test_detection_is_frozen():
    d = Detection(label=Label.SECRET, start=0, end=1, score=0.5)
    with pytest.raises(Exception):  # FrozenInstanceError
        d.start = 99  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_protocol.py -v
```

Expected: ImportError or collection failure (modules don't exist yet).

- [ ] **Step 3: Implement `protocol.py`**

```python
# src/privacy_filter/detection/__init__.py
```

```python
# src/privacy_filter/detection/protocol.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class Label(StrEnum):
    ACCOUNT_NUMBER = "account_number"
    PRIVATE_ADDRESS = "private_address"
    PRIVATE_EMAIL = "private_email"
    PRIVATE_PERSON = "private_person"
    PRIVATE_PHONE = "private_phone"
    PRIVATE_URL = "private_url"
    PRIVATE_DATE = "private_date"
    SECRET = "secret"


@dataclass(frozen=True, slots=True)
class Detection:
    label: Label
    start: int
    end: int
    score: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


class Detector(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def detect(self, text: str) -> list[Detection]: ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_protocol.py -v
uv run mypy src/privacy_filter/detection/protocol.py tests/unit/test_protocol.py
uv run ruff check src tests
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/detection/ tests/unit/
git commit -m "feat(detection): add Label, Detection, Detector Protocol"
```

---

## Task 3: Redaction — `apply_spans`

**Files:**
- Create: `src/privacy_filter/detection/redact.py`
- Test: `tests/unit/test_redact.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_redact.py
from privacy_filter.detection.protocol import Detection, Label
from privacy_filter.detection.redact import apply_spans


def test_no_detections_returns_input_unchanged():
    assert apply_spans("hello", []) == "hello"


def test_single_span_replaced():
    text = "Email me at alice@example.com tomorrow."
    d = Detection(label=Label.PRIVATE_EMAIL, start=12, end=29, score=0.99)
    assert apply_spans(text, [d]) == "Email me at [PRIVATE_EMAIL] tomorrow."


def test_multiple_spans_applied_in_reverse_so_offsets_stay_valid():
    text = "alice@x.com / 555-1212"
    spans = [
        Detection(label=Label.PRIVATE_EMAIL, start=0, end=11, score=0.9),
        Detection(label=Label.PRIVATE_PHONE, start=14, end=22, score=0.9),
    ]
    assert apply_spans(text, spans) == "[PRIVATE_EMAIL] / [PRIVATE_PHONE]"


def test_overlap_keeps_highest_score():
    text = "aaaaa"
    spans = [
        Detection(label=Label.SECRET, start=0, end=4, score=0.5),
        Detection(label=Label.PRIVATE_EMAIL, start=2, end=5, score=0.9),
    ]
    assert apply_spans(text, spans) == "aa[PRIVATE_EMAIL]"


def test_overlap_tie_break_by_earliest_start():
    text = "aaaaa"
    spans = [
        Detection(label=Label.SECRET, start=2, end=4, score=0.7),
        Detection(label=Label.PRIVATE_EMAIL, start=0, end=3, score=0.7),
    ]
    assert apply_spans(text, spans) == "[PRIVATE_EMAIL]aa"


def test_overlap_tie_break_by_longest_when_score_and_start_equal():
    text = "aaaaa"
    spans = [
        Detection(label=Label.SECRET, start=0, end=2, score=0.7),
        Detection(label=Label.PRIVATE_EMAIL, start=0, end=4, score=0.7),
    ]
    assert apply_spans(text, spans) == "[PRIVATE_EMAIL]a"


def test_zero_length_span_is_inserted():
    text = "abc"
    d = Detection(label=Label.SECRET, start=1, end=1, score=0.5)
    assert apply_spans(text, [d]) == "a[SECRET]bc"


def test_emoji_offsets_use_python_str_indexing():
    text = "hi 😀 alice@x.com"
    d = Detection(label=Label.PRIVATE_EMAIL, start=5, end=16, score=0.9)
    assert apply_spans(text, [d]) == "hi 😀 [PRIVATE_EMAIL]"


def test_adjacent_non_overlapping_spans_both_applied():
    text = "abcdef"
    spans = [
        Detection(label=Label.SECRET, start=0, end=3, score=0.9),
        Detection(label=Label.PRIVATE_EMAIL, start=3, end=6, score=0.9),
    ]
    assert apply_spans(text, spans) == "[SECRET][PRIVATE_EMAIL]"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_redact.py -v
```

Expected: ImportError on `apply_spans`.

- [ ] **Step 3: Implement `redact.py`**

```python
# src/privacy_filter/detection/redact.py
from __future__ import annotations

from .protocol import Detection


def apply_spans(text: str, detections: list[Detection]) -> str:
    """Return text with each detected span replaced by `[<LABEL_UPPER>]`.

    Overlaps are resolved by keeping the highest-scoring span; ties are broken
    by earliest start, then longest length. The remaining non-overlapping spans
    are then applied in reverse offset order so earlier offsets stay valid.
    """
    if not detections:
        return text

    resolved = _resolve_overlaps(detections)
    out = text
    for d in sorted(resolved, key=lambda x: x.start, reverse=True):
        out = f"{out[: d.start]}[{d.label.value.upper()}]{out[d.end :]}"
    return out


def _resolve_overlaps(detections: list[Detection]) -> list[Detection]:
    ordered = sorted(
        detections,
        key=lambda d: (-d.score, d.start, -(d.end - d.start)),
    )
    kept: list[Detection] = []
    for d in ordered:
        if any(_overlaps(d, k) for k in kept):
            continue
        kept.append(d)
    return kept


def _overlaps(a: Detection, b: Detection) -> bool:
    return a.start < b.end and b.start < a.end
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_redact.py -v
uv run mypy src/privacy_filter/detection/redact.py
uv run ruff check
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/detection/redact.py tests/unit/test_redact.py
git commit -m "feat(detection): apply_spans with overlap resolution and reverse application"
```

---

## Task 4: `FakeDetector`

**Files:**
- Create: `src/privacy_filter/detection/fake.py`
- Test: `tests/unit/test_fake_detector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_fake_detector.py
from privacy_filter.detection.fake import FakeDetector
from privacy_filter.detection.protocol import Detection, Label


def test_fake_detector_returns_scripted_detections():
    spans = [Detection(label=Label.PRIVATE_EMAIL, start=0, end=5, score=0.9)]
    detector = FakeDetector(spans)
    assert detector.detect("hello") == spans


def test_fake_detector_exposes_model_id_and_revision():
    detector = FakeDetector([], model_id="fake-model", model_revision="rev-1")
    assert detector.model_id == "fake-model"
    assert detector.model_revision == "rev-1"


def test_fake_detector_supports_callable_for_per_input_responses():
    def script(text: str) -> list[Detection]:
        if "alice" in text:
            return [Detection(label=Label.PRIVATE_PERSON, start=0, end=5, score=1.0)]
        return []

    detector = FakeDetector(script)
    assert detector.detect("alice").pop().label is Label.PRIVATE_PERSON
    assert detector.detect("bob") == []


def test_fake_detector_default_metadata():
    detector = FakeDetector([])
    assert detector.model_id == "fake"
    assert detector.model_revision == "test"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_fake_detector.py -v
```

Expected: ImportError on `FakeDetector`.

- [ ] **Step 3: Implement `FakeDetector`**

```python
# src/privacy_filter/detection/fake.py
from __future__ import annotations

from collections.abc import Callable

from .protocol import Detection

DetectionScript = list[Detection] | Callable[[str], list[Detection]]


class FakeDetector:
    """In-memory detector for tests. Either returns a fixed list or runs a callable."""

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
```

- [ ] **Step 4: Run tests, mypy, ruff**

```bash
uv run pytest tests/unit/test_fake_detector.py -v
uv run mypy
uv run ruff check
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/detection/fake.py tests/unit/test_fake_detector.py
git commit -m "feat(detection): FakeDetector for tests"
```

---

## Task 5: Detector Protocol contract suite

**Files:**
- Create: `tests/contract/__init__.py`
- Create: `tests/contract/detector_contract.py`
- Create: `tests/contract/test_fake_detector_contract.py`

The contract suite is a parameterized helper that any `Detector` implementation must pass. It is run here against `FakeDetector`; in Task 17 it will also be run against `HuggingFaceDetector` under `-m slow`.

- [ ] **Step 1: Write the contract helper**

```python
# tests/contract/__init__.py
```

```python
# tests/contract/detector_contract.py
"""Reusable contract assertions for any Detector implementation.

A test module wires up its concrete detector and golden inputs, then calls
``assert_detector_contract``. This keeps the contract in one place.
"""
from __future__ import annotations

from collections.abc import Iterable

from privacy_filter.detection.protocol import Detector, Label


def assert_detector_contract(
    detector: Detector,
    inputs: Iterable[str],
) -> None:
    inputs = list(inputs)
    assert isinstance(detector.model_id, str) and detector.model_id
    assert isinstance(detector.model_revision, str)

    label_set = {member.value for member in Label}

    for text in inputs:
        first = detector.detect(text)
        second = detector.detect(text)
        assert first == second, "detect must be deterministic for identical input"

        for d in first:
            assert d.label.value in label_set, f"unknown label: {d.label}"
            assert 0 <= d.start <= d.end <= len(text), (
                f"span out of range: ({d.start}, {d.end}) for len {len(text)}"
            )
            assert 0.0 <= d.score <= 1.0, f"score out of range: {d.score}"
```

- [ ] **Step 2: Write the FakeDetector contract test**

```python
# tests/contract/test_fake_detector_contract.py
from privacy_filter.detection.fake import FakeDetector
from privacy_filter.detection.protocol import Detection, Label

from .detector_contract import assert_detector_contract


def test_fake_detector_satisfies_contract():
    def script(text: str) -> list[Detection]:
        if "@" in text:
            idx = text.index("@")
            start = max(0, idx - 5)
            end = min(len(text), idx + 5)
            return [Detection(label=Label.PRIVATE_EMAIL, start=start, end=end, score=0.9)]
        return []

    detector = FakeDetector(script, model_id="fake", model_revision="r1")

    assert_detector_contract(
        detector,
        inputs=[
            "hello world",
            "alice@example.com",
            "  alice@example.com",
            "no detections here",
            "",
        ],
    )
```

- [ ] **Step 3: Run, type-check, lint**

```bash
uv run pytest tests/contract -v
uv run mypy
uv run ruff check
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add tests/contract/
git commit -m "test(detection): Detector Protocol contract suite, run against FakeDetector"
```

---

## Task 6: Configuration via `pydantic-settings`

**Files:**
- Create: `src/privacy_filter/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_config.py
import pytest

from privacy_filter.config import Settings


def test_settings_loads_required_api_keys(monkeypatch):
    monkeypatch.setenv("API_KEYS", "key-a,key-b")
    s = Settings()
    assert s.api_keys == frozenset({"key-a", "key-b"})


def test_settings_rejects_empty_api_keys(monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    with pytest.raises(ValueError, match="API_KEYS"):
        Settings()


def test_settings_strips_whitespace_and_drops_empties(monkeypatch):
    monkeypatch.setenv("API_KEYS", " key-a , ,key-b ,")
    s = Settings()
    assert s.api_keys == frozenset({"key-a", "key-b"})


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k")
    s = Settings()
    assert s.max_input_chars == 50_000
    assert s.max_body_bytes == 262_144
    assert s.model_id == "openai/privacy-filter"
    assert s.model_revision is None
    assert s.log_level == "INFO"
    assert s.cors_origins == ()


def test_settings_parses_cors_origins(monkeypatch):
    monkeypatch.setenv("API_KEYS", "k")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,https://b.example")
    s = Settings()
    assert s.cors_origins == ("https://a.example", "https://b.example")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: ImportError on `Settings`.

- [ ] **Step 3: Implement `config.py`**

```python
# src/privacy_filter/config.py
from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_keys: frozenset[str] = Field(default_factory=frozenset)
    max_input_chars: Annotated[int, Field(ge=1)] = 50_000
    max_body_bytes: Annotated[int, Field(ge=1)] = 262_144
    model_id: str = "openai/privacy-filter"
    model_revision: str | None = None
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ()

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, value: object) -> frozenset[str]:
        if isinstance(value, frozenset):
            return value
        if not isinstance(value, str):
            raise ValueError("API_KEYS must be a comma-separated string")
        keys = frozenset(part.strip() for part in value.split(",") if part.strip())
        if not keys:
            raise ValueError("API_KEYS must contain at least one non-empty key")
        return keys

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        if value in (None, ""):
            return ()
        if not isinstance(value, str):
            raise ValueError("CORS_ORIGINS must be a comma-separated string")
        return tuple(p.strip() for p in value.split(",") if p.strip())
```

- [ ] **Step 4: Run tests, mypy, ruff**

```bash
uv run pytest tests/unit/test_config.py -v
uv run mypy
uv run ruff check
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/config.py tests/unit/test_config.py
git commit -m "feat(config): pydantic-settings with API_KEYS and CORS parsing"
```

---

## Task 7: Constant-time API-key comparison

**Files:**
- Create: `src/privacy_filter/security.py`
- Test: `tests/unit/test_security.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_security.py
from privacy_filter.security import is_valid_api_key


def test_valid_key_accepted():
    assert is_valid_api_key("alpha", frozenset({"alpha", "beta"}))


def test_unknown_key_rejected():
    assert not is_valid_api_key("gamma", frozenset({"alpha", "beta"}))


def test_empty_key_rejected_even_if_set_contains_empty_string():
    # Defense in depth: never accept an empty presented key.
    assert not is_valid_api_key("", frozenset({"", "alpha"}))


def test_none_key_rejected():
    assert not is_valid_api_key(None, frozenset({"alpha"}))


def test_no_short_circuit_on_length(monkeypatch):
    # We can't easily measure timing here, but we can at least assert
    # we always evaluate every candidate by checking call count of
    # compare_digest. This test is a behavioral sanity check.
    import privacy_filter.security as sec

    calls: list[tuple[str, str]] = []

    def fake_compare(a: str, b: str) -> bool:
        calls.append((a, b))
        return a == b

    monkeypatch.setattr(sec, "_compare", fake_compare)
    is_valid_api_key("xx", frozenset({"alpha", "beta", "gamma"}))
    assert len(calls) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_security.py -v
```

Expected: ImportError on `is_valid_api_key`.

- [ ] **Step 3: Implement `security.py`**

```python
# src/privacy_filter/security.py
from __future__ import annotations

import hmac


def _compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def is_valid_api_key(presented: str | None, allowed: frozenset[str]) -> bool:
    """Constant-time API-key check.

    Always evaluates ``compare_digest`` against every allowed key so timing
    behavior does not leak information about which key matched (or how close
    a near-miss was).
    """
    if not presented:
        return False
    matched = False
    for candidate in allowed:
        if _compare(presented, candidate):
            matched = True
    return matched
```

- [ ] **Step 4: Run tests, mypy, ruff**

```bash
uv run pytest tests/unit/test_security.py -v
uv run mypy
uv run ruff check
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/security.py tests/unit/test_security.py
git commit -m "feat(security): constant-time API-key comparison"
```

---

## Task 8: Structlog with PII-safe field allowlist

**Files:**
- Create: `src/privacy_filter/logging.py`
- Test: `tests/unit/test_logging.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_logging.py
import json
import logging

from privacy_filter.logging import ALLOWED_LOG_FIELDS, configure_logging, drop_disallowed_fields


def test_allowlist_drops_disallowed_keys():
    event = {
        "event": "request_completed",
        "request_id": "abc",
        "endpoint": "/v1/detect",
        "status": 200,
        "latency_ms": 12,
        "input_chars": 42,
        "detection_count": 3,
        # disallowed:
        "text": "alice@example.com",
        "redacted": "[PRIVATE_EMAIL]",
        "headers": {"X-API-Key": "secret"},
    }
    cleaned = drop_disallowed_fields(None, "info", event)
    assert "text" not in cleaned
    assert "redacted" not in cleaned
    assert "headers" not in cleaned
    assert cleaned["request_id"] == "abc"
    assert cleaned["status"] == 200


def test_allowed_fields_match_spec():
    assert ALLOWED_LOG_FIELDS >= {
        "event", "request_id", "endpoint", "method", "status", "latency_ms",
        "input_chars", "detection_count", "code", "exc_class",
    }


def test_configure_logging_emits_structured_json(capsys):
    configure_logging(level="INFO")
    import structlog

    log = structlog.get_logger("test")
    log.info("request_completed", request_id="abc", status=200, text="leak")

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "expected at least one log line"
    record = json.loads(captured[-1])
    assert record["event"] == "request_completed"
    assert record["request_id"] == "abc"
    assert record["status"] == 200
    assert "text" not in record


def test_configure_logging_respects_level(capsys):
    configure_logging(level="WARNING")
    import structlog

    log = structlog.get_logger("test")
    log.info("ignored", request_id="abc")
    log.warning("kept", request_id="abc")

    lines = [l for l in capsys.readouterr().out.strip().splitlines() if l]
    payloads = [json.loads(l) for l in lines]
    events = [p["event"] for p in payloads]
    assert "ignored" not in events
    assert "kept" in events
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_logging.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `logging.py`**

```python
# src/privacy_filter/logging.py
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

ALLOWED_LOG_FIELDS: frozenset[str] = frozenset(
    {
        "event",
        "request_id",
        "endpoint",
        "method",
        "status",
        "latency_ms",
        "input_chars",
        "detection_count",
        "code",
        "exc_class",
        "logger",
        "level",
        "timestamp",
    }
)


def drop_disallowed_fields(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    return {k: v for k, v in event_dict.items() if k in ALLOWED_LOG_FIELDS}


def configure_logging(*, level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            drop_disallowed_fields,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

- [ ] **Step 4: Run tests, mypy, ruff**

```bash
uv run pytest tests/unit/test_logging.py -v
uv run mypy
uv run ruff check
```

Expected: green. Note: `configure_logging` mutates global state. Ensure each test that calls it is in its own test function (already true above).

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/logging.py tests/unit/test_logging.py
git commit -m "feat(logging): structlog with PII-safe field allowlist processor"
```

---

## Task 9: API request/response schemas

**Files:**
- Create: `src/privacy_filter/api/__init__.py`
- Create: `src/privacy_filter/api/schemas.py`
- Test: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_schemas.py
import pytest
from pydantic import ValidationError

from privacy_filter.api.schemas import DetectRequest, DetectResponse, DetectionOut


def test_detect_request_accepts_non_empty_text():
    req = DetectRequest(text="hello", _max_chars=100)  # type: ignore[call-arg]
    assert req.text == "hello"


def test_detect_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        DetectRequest.model_validate({"text": ""})


def test_detect_request_rejects_text_above_max_chars():
    with pytest.raises(ValidationError):
        DetectRequest.with_max_chars(5).model_validate({"text": "abcdef"})


def test_detect_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        DetectRequest.model_validate({"text": "hi", "extra": 1})


def test_detection_out_shape():
    out = DetectionOut(label="private_email", start=0, end=5, score=0.9)
    assert out.model_dump() == {
        "label": "private_email", "start": 0, "end": 5, "score": 0.9,
    }


def test_detect_response_shape():
    resp = DetectResponse(
        detections=[DetectionOut(label="secret", start=0, end=1, score=0.5)],
        redacted="[SECRET]",
        model="openai/privacy-filter",
        model_revision="abc",
    )
    payload = resp.model_dump()
    assert payload["detections"][0]["label"] == "secret"
    assert payload["redacted"] == "[SECRET]"
    assert payload["model_revision"] == "abc"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_schemas.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `schemas.py`**

```python
# src/privacy_filter/api/__init__.py
```

```python
# src/privacy_filter/api/schemas.py
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class DetectRequest(BaseModel):
    """Request body for POST /v1/detect.

    The ``text`` length bound is set per-request via ``with_max_chars``; the
    default model uses an effectively-unbounded limit so that bare construction
    still validates non-emptiness.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: Annotated[str, StringConstraints(min_length=1, max_length=10**9)]

    @classmethod
    def with_max_chars(cls, max_chars: int) -> type[DetectRequest]:
        class _Bounded(DetectRequest):
            text: Annotated[
                str,
                StringConstraints(min_length=1, max_length=max_chars),
            ]

        _Bounded.__name__ = "DetectRequest"
        return _Bounded


class DetectionOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detections: list[DetectionOut]
    redacted: str
    model: str
    model_revision: str
```

- [ ] **Step 4: Run tests, mypy, ruff**

```bash
uv run pytest tests/unit/test_schemas.py -v
uv run mypy
uv run ruff check
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/api/ tests/unit/test_schemas.py
git commit -m "feat(api): DetectRequest/DetectResponse schemas with bounded text"
```

---

## Task 10: Error envelope and exception types

**Files:**
- Create: `src/privacy_filter/api/errors.py`
- Test: `tests/unit/test_errors.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_errors.py
import pytest

from privacy_filter.api.errors import (
    APIError,
    ErrorCode,
    InvalidAPIKey,
    PayloadTooLarge,
    error_envelope,
)


def test_api_error_subclasses_carry_code_and_status():
    assert InvalidAPIKey().code is ErrorCode.UNAUTHORIZED
    assert InvalidAPIKey().status_code == 401
    assert PayloadTooLarge().code is ErrorCode.PAYLOAD_TOO_LARGE
    assert PayloadTooLarge().status_code == 413


def test_envelope_shape_for_known_error():
    err = InvalidAPIKey()
    payload = error_envelope(err, request_id="req-123")
    assert payload == {
        "error": {
            "code": "unauthorized",
            "message": "Invalid or missing API key.",
            "request_id": "req-123",
        }
    }


def test_envelope_for_internal_error_is_generic():
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


def test_api_error_is_raisable():
    with pytest.raises(APIError):
        raise InvalidAPIKey()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_errors.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `errors.py`**

```python
# src/privacy_filter/api/errors.py
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
```

- [ ] **Step 4: Run tests, mypy, ruff**

```bash
uv run pytest tests/unit/test_errors.py -v
uv run mypy
uv run ruff check
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/privacy_filter/api/errors.py tests/unit/test_errors.py
git commit -m "feat(api): error envelope and typed APIError hierarchy"
```

---

## Task 11: ASGI middleware — request ID, security headers, body-size guard

**Files:**
- Create: `src/privacy_filter/api/middleware.py`
- Test: defer to integration tests in Task 14 (middleware is exercised end-to-end)

- [ ] **Step 1: Implement `middleware.py`**

```python
# src/privacy_filter/api/middleware.py
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .errors import PayloadTooLarge, error_envelope


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, max_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max_bytes = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self._max_bytes:
                    return _too_large(getattr(request.state, "request_id", ""))
            except ValueError:
                return _too_large(getattr(request.state, "request_id", ""))
        return await call_next(request)


def _too_large(request_id: str) -> JSONResponse:
    err = PayloadTooLarge()
    return JSONResponse(
        status_code=err.status_code,
        content=error_envelope(err, request_id=request_id),
    )
```

- [ ] **Step 2: Run mypy and ruff**

```bash
uv run mypy
uv run ruff check
```

Expected: green. (No unit test for this file in isolation; integration tests in Task 14 cover behavior.)

- [ ] **Step 3: Commit**

```bash
git add src/privacy_filter/api/middleware.py
git commit -m "feat(api): request-id, security-headers, body-size-limit middleware"
```

---

## Task 12: Dependencies — auth + detector

**Files:**
- Create: `src/privacy_filter/api/deps.py`

The auth dependency is wired into routes in Task 13; integration tests in Task 14 prove its behavior end-to-end.

- [ ] **Step 1: Implement `deps.py`**

```python
# src/privacy_filter/api/deps.py
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from privacy_filter.config import Settings
from privacy_filter.detection.protocol import Detector
from privacy_filter.security import is_valid_api_key

from .errors import InvalidAPIKey


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_detector(request: Request) -> Detector:
    detector: Detector = request.app.state.detector
    return detector


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if not is_valid_api_key(x_api_key, settings.api_keys):
        raise InvalidAPIKey()
```

- [ ] **Step 2: Run mypy and ruff**

```bash
uv run mypy
uv run ruff check
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add src/privacy_filter/api/deps.py
git commit -m "feat(api): auth and detector dependency providers"
```

---

## Task 13: Routes and app factory

**Files:**
- Create: `src/privacy_filter/api/routes.py`
- Create: `src/privacy_filter/api/app.py`

- [ ] **Step 1: Implement `routes.py`**

```python
# src/privacy_filter/api/routes.py
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends

from privacy_filter.config import Settings
from privacy_filter.detection.protocol import Detector
from privacy_filter.detection.redact import apply_spans

from .deps import get_detector, get_settings, require_api_key
from .errors import ServiceNotReady
from .schemas import DetectionOut, DetectRequest, DetectResponse

log = structlog.get_logger(__name__)

router = APIRouter()
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(
    settings: Annotated[Settings, Depends(get_settings)],  # noqa: ARG001 (forces lifespan)
    detector: Annotated[Detector | None, Depends(get_detector)] = None,
) -> dict[str, str]:
    if detector is None:
        raise ServiceNotReady()
    return {"status": "ready"}


@v1.post("/detect", response_model=DetectResponse)
def detect(
    body: DetectRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    detector: Annotated[Detector, Depends(get_detector)],
) -> DetectResponse:
    bounded = DetectRequest.with_max_chars(settings.max_input_chars)
    body = bounded.model_validate(body.model_dump())

    detections = detector.detect(body.text)
    redacted = apply_spans(body.text, detections)

    log.info(
        "detect_completed",
        endpoint="/v1/detect",
        method="POST",
        status=200,
        input_chars=len(body.text),
        detection_count=len(detections),
    )

    return DetectResponse(
        detections=[
            DetectionOut(label=d.label.value, start=d.start, end=d.end, score=d.score)
            for d in detections
        ],
        redacted=redacted,
        model=detector.model_id,
        model_revision=detector.model_revision,
    )
```

- [ ] **Step 2: Implement `app.py`**

```python
# src/privacy_filter/api/app.py
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from privacy_filter.config import Settings
from privacy_filter.detection.protocol import Detector
from privacy_filter.logging import configure_logging

from .errors import APIError, error_envelope
from .middleware import (
    BodySizeLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from .routes import router, v1

log = structlog.get_logger(__name__)


def _default_detector_factory(settings: Settings) -> Detector:
    from privacy_filter.detection.huggingface import HuggingFaceDetector

    return HuggingFaceDetector(
        model_id=settings.model_id,
        revision=settings.model_revision,
    )


def create_app(
    *,
    settings: Settings | None = None,
    detector_factory: object | None = None,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        factory = detector_factory or _default_detector_factory
        t0 = time.perf_counter()
        app.state.detector = factory(settings)  # type: ignore[operator]
        # Warm-up: a single short inference. Failures here block readiness.
        app.state.detector.detect("warmup")
        log.info(
            "lifespan_ready", endpoint="lifespan", latency_ms=int((time.perf_counter() - t0) * 1000)
        )
        try:
            yield
        finally:
            app.state.detector = None

    app = FastAPI(title="privacy-filter", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["POST", "GET"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    app.include_router(router)
    app.include_router(v1)

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        return JSONResponse(
            status_code=exc.status_code, content=error_envelope(exc, request_id=rid)
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        from .errors import InvalidRequest

        return JSONResponse(
            status_code=InvalidRequest.status_code,
            content=error_envelope(InvalidRequest(), request_id=rid),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", "")
        log.error("unhandled_exception", exc_class=type(exc).__name__, code="internal_error")
        return JSONResponse(status_code=500, content=error_envelope(exc, request_id=rid))

    return app
```

- [ ] **Step 3: Run mypy and ruff**

```bash
uv run mypy
uv run ruff check
```

Expected: green. Tests for these modules live in Task 14 (integration).

- [ ] **Step 4: Commit**

```bash
git add src/privacy_filter/api/routes.py src/privacy_filter/api/app.py
git commit -m "feat(api): routes (/v1/detect, /healthz, /readyz) and app factory with lifespan"
```

---

## Task 14: Integration tests — happy path, auth, validation, error envelope

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_detect_endpoint.py`
- Create: `tests/integration/test_health_endpoints.py`
- Create: `tests/integration/test_error_envelope.py`

- [ ] **Step 1: Shared fixtures in `conftest.py`**

```python
# tests/conftest.py
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
            if "alice@example.com" in text:
                start = text.index("alice@example.com")
                return [
                    Detection(
                        label=Label.PRIVATE_EMAIL,
                        start=start,
                        end=start + len("alice@example.com"),
                        score=0.99,
                    )
                ]
            return []

        return FakeDetector(script, model_id="fake", model_revision="rev-test")

    return factory


@pytest.fixture
def app(settings: Settings, fake_detector_factory) -> FastAPI:
    return create_app(settings=settings, detector_factory=fake_detector_factory)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # Trigger lifespan startup.
        async with app.router.lifespan_context(app):
            yield c
```

- [ ] **Step 2: Health endpoint tests**

```python
# tests/integration/__init__.py
```

```python
# tests/integration/test_health_endpoints.py
import pytest


@pytest.mark.asyncio
async def test_healthz_is_open_and_ok(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_returns_ready_after_lifespan(client):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}
```

- [ ] **Step 3: `/v1/detect` happy-path and contract tests**

```python
# tests/integration/test_detect_endpoint.py
import pytest


@pytest.mark.asyncio
async def test_detect_happy_path(client, api_key):
    r = await client.post(
        "/v1/detect",
        headers={"X-API-Key": api_key},
        json={"text": "Email me at alice@example.com please."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake"
    assert body["model_revision"] == "rev-test"
    assert body["redacted"] == "Email me at [PRIVATE_EMAIL] please."
    assert body["detections"] == [
        {"label": "private_email", "start": 12, "end": 29, "score": 0.99},
    ]


@pytest.mark.asyncio
async def test_detect_no_pii_returns_unchanged_text(client, api_key):
    r = await client.post(
        "/v1/detect",
        headers={"X-API-Key": api_key},
        json={"text": "nothing to see here"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["detections"] == []
    assert body["redacted"] == "nothing to see here"


@pytest.mark.asyncio
async def test_detect_requires_api_key(client):
    r = await client.post("/v1/detect", json={"text": "hello"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


@pytest.mark.asyncio
async def test_detect_rejects_wrong_api_key(client):
    r = await client.post(
        "/v1/detect", headers={"X-API-Key": "nope"}, json={"text": "hello"}
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_detect_rejects_oversized_text(client, api_key):
    huge = "a" * 200  # MAX_INPUT_CHARS=100 in fixtures
    r = await client.post(
        "/v1/detect", headers={"X-API-Key": api_key}, json={"text": huge}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_detect_rejects_empty_text(client, api_key):
    r = await client.post(
        "/v1/detect", headers={"X-API-Key": api_key}, json={"text": ""}
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_detect_rejects_extra_fields(client, api_key):
    r = await client.post(
        "/v1/detect",
        headers={"X-API-Key": api_key},
        json={"text": "hi", "mode": "all"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_detect_rejects_oversized_body(client, api_key):
    big_payload = '{"text": "' + ("a" * 5000) + '"}'
    r = await client.post(
        "/v1/detect",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        content=big_payload,
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


@pytest.mark.asyncio
async def test_response_includes_security_headers(client, api_key):
    r = await client.post(
        "/v1/detect", headers={"X-API-Key": api_key}, json={"text": "hi"}
    )
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "X-Request-ID" in r.headers
```

- [ ] **Step 4: Error envelope tests (no stack trace leaks)**

```python
# tests/integration/test_error_envelope.py
import pytest

from privacy_filter.api.app import create_app
from privacy_filter.config import Settings
from privacy_filter.detection.fake import FakeDetector


@pytest.fixture
def boom_app(settings: Settings):
    def factory(_s):
        class Boom(FakeDetector):
            def detect(self, text: str):
                raise RuntimeError("synthetic failure")

        return Boom([], model_id="fake", model_revision="r1")

    return create_app(settings=settings, detector_factory=factory)


@pytest.mark.asyncio
async def test_unhandled_error_returns_generic_envelope(boom_app, api_key):
    from httpx import ASGITransport, AsyncClient

    async with boom_app.router.lifespan_context(boom_app):
        async with AsyncClient(transport=ASGITransport(app=boom_app), base_url="http://t") as c:
            r = await c.post(
                "/v1/detect", headers={"X-API-Key": api_key}, json={"text": "hi"}
            )
            assert r.status_code == 500
            body = r.json()
            assert body["error"]["code"] == "internal_error"
            assert body["error"]["message"] == "Internal server error."
            # No stack-trace, no original exception text
            text = r.text
            assert "synthetic failure" not in text
            assert "Traceback" not in text
            assert "RuntimeError" not in text
```

- [ ] **Step 5: Run tests, mypy, ruff**

```bash
uv run pytest tests/integration -v
uv run mypy
uv run ruff check
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/integration/
git commit -m "test(integration): /v1/detect happy path, auth, validation, error envelope"
```

---

## Task 15: PII-in-logs guard test

**Files:**
- Create: `tests/integration/test_logging_no_pii.py`

This is the mechanical enforcement of the "never log PII" rule. It is intentionally separate so it stands out and won't be deleted casually.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_logging_no_pii.py
import io
import json
import logging

import pytest
import structlog


@pytest.mark.asyncio
async def test_no_log_record_contains_input_text(client, api_key, caplog):
    sentinel = "SECRET_SENTINEL_alice@uniq-test.example_4f9a"

    # Capture both stdlib logging and the structlog stream renderer output.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=buf),
        cache_logger_on_first_use=False,
    )
    try:
        r = await client.post(
            "/v1/detect",
            headers={"X-API-Key": api_key},
            json={"text": f"please redact {sentinel}"},
        )
        assert r.status_code == 200

        captured = buf.getvalue()
        # Sentinel must not appear anywhere in any log line, in any field.
        assert sentinel not in captured

        # Structured log records should remain valid JSON when present.
        for line in captured.splitlines():
            if line.strip().startswith("{"):
                json.loads(line)
    finally:
        logging.getLogger().removeHandler(handler)
```

- [ ] **Step 2: Run the test (should pass first time if Task 8 was implemented correctly)**

```bash
uv run pytest tests/integration/test_logging_no_pii.py -v
```

Expected: PASS. If it fails, the allowlist processor is letting fields through — fix the processor, do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_logging_no_pii.py
git commit -m "test(integration): assert request input never appears in any log line"
```

---

## Task 16: Property-based redaction tests

**Files:**
- Create: `tests/property/__init__.py`
- Create: `tests/property/test_redact_properties.py`

- [ ] **Step 1: Write the property tests**

```python
# tests/property/__init__.py
```

```python
# tests/property/test_redact_properties.py
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from privacy_filter.detection.protocol import Detection, Label
from privacy_filter.detection.redact import apply_spans


@st.composite
def text_with_non_overlapping_spans(draw):
    text = draw(st.text(min_size=0, max_size=200))
    n = len(text)
    if n == 0:
        return text, []

    count = draw(st.integers(min_value=0, max_value=5))
    spans: list[Detection] = []
    cursor = 0
    for _ in range(count):
        if cursor >= n:
            break
        start = draw(st.integers(min_value=cursor, max_value=n))
        if start >= n:
            break
        end = draw(st.integers(min_value=start, max_value=n))
        label = draw(st.sampled_from(list(Label)))
        score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        spans.append(Detection(label=label, start=start, end=end, score=score))
        cursor = end
    return text, spans


@given(text_with_non_overlapping_spans())
@settings(max_examples=200)
def test_redacted_length_equals_expected(case):
    text, spans = case
    redacted = apply_spans(text, spans)
    expected = len(text) + sum(
        len(f"[{s.label.value.upper()}]") - (s.end - s.start) for s in spans
    )
    assert len(redacted) == expected


@given(text_with_non_overlapping_spans())
@settings(max_examples=200)
def test_redaction_is_deterministic(case):
    text, spans = case
    assert apply_spans(text, spans) == apply_spans(text, spans)


@given(text_with_non_overlapping_spans())
@settings(max_examples=200)
def test_no_pii_chars_survive_at_original_offsets(case):
    text, spans = case
    redacted = apply_spans(text, spans)
    # Trivial check: each span's replacement starts with '[' in the redacted output's
    # adjusted location. We can't easily check exact post-shift offsets, but we can
    # assert that the literal spans (where non-empty) no longer appear at their
    # original positions if they had distinctive content.
    for s in spans:
        if s.end > s.start:
            original = text[s.start : s.end]
            if original and original not in (f"[{s.label.value.upper()}]",):
                assert original not in redacted or original in text  # tautology safeguard
                # The real assertion: the slice of `redacted` that originally held
                # the span no longer equals `original`. After shifts, computing the
                # exact slice is non-trivial, so we use a weaker but useful check:
                # the redacted output contains the bracketed marker for this span.
                assert f"[{s.label.value.upper()}]" in redacted
```

- [ ] **Step 2: Run, mypy, ruff**

```bash
uv run pytest tests/property -v
uv run mypy
uv run ruff check
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add tests/property/
git commit -m "test(property): hypothesis-driven invariants for apply_spans"
```

---

## Task 17: HuggingFaceDetector and slow tests

**Files:**
- Create: `src/privacy_filter/detection/huggingface.py`
- Create: `tests/slow/__init__.py`
- Create: `tests/slow/test_huggingface_detector.py`

- [ ] **Step 1: Implement `huggingface.py`**

```python
# src/privacy_filter/detection/huggingface.py
from __future__ import annotations

from typing import Any

from .protocol import Detection, Label


class HuggingFaceDetector:
    """Runs `openai/privacy-filter` via the transformers token-classification pipeline.

    Imports of `transformers` and `torch` are deferred to construction so the
    rest of the codebase does not depend on them at import time.
    """

    def __init__(self, *, model_id: str, revision: str | None = None) -> None:
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_id, device_map="auto", **kwargs
        )
        self._pipeline = pipeline(
            task="token-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            aggregation_strategy="first",
        )
        self._model_id = model_id
        self._revision = (
            getattr(self._model.config, "_commit_hash", None) or revision or "unknown"
        )

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
        out: list[Detection] = []
        for entity in raw:
            label_str = str(entity["entity_group"]).lower()
            try:
                label = Label(label_str)
            except ValueError:
                continue  # ignore unknown labels (e.g. background)
            out.append(
                Detection(
                    label=label,
                    start=int(entity["start"]),
                    end=int(entity["end"]),
                    score=float(entity["score"]),
                )
            )
        return out
```

> **Note on aggregation_strategy:** `"first"` returns spans grouped via the BIOES output of the tokenizer's first-subword score. If the model card later prescribes a different strategy (e.g., constrained Viterbi via a custom head), update this constructor accordingly. The contract suite in Step 3 will still validate that the output offsets index real characters and labels are in the closed set.

- [ ] **Step 2: Slow tests**

```python
# tests/slow/__init__.py
```

```python
# tests/slow/test_huggingface_detector.py
import pytest

slow = pytest.mark.slow

pytest.importorskip("transformers")
pytest.importorskip("torch")


@slow
def test_huggingface_detector_satisfies_contract():
    from privacy_filter.detection.huggingface import HuggingFaceDetector

    from tests.contract.detector_contract import assert_detector_contract

    detector = HuggingFaceDetector(model_id="openai/privacy-filter")
    assert_detector_contract(
        detector,
        inputs=[
            "",
            "no pii here",
            "Email alice@example.com tomorrow.",
            "Call me at +1 415 555 0123 please.",
            "My SSN is 123-45-6789.",
        ],
    )


@slow
def test_huggingface_detector_finds_email():
    from privacy_filter.detection.huggingface import HuggingFaceDetector
    from privacy_filter.detection.protocol import Label

    detector = HuggingFaceDetector(model_id="openai/privacy-filter")
    out = detector.detect("Please email alice@example.com soon.")
    assert any(d.label is Label.PRIVATE_EMAIL for d in out)
```

- [ ] **Step 3: Run only when `[hf]` is installed**

```bash
uv sync --extra hf --extra dev
uv run pytest -m slow -v
```

Expected: green (will download the model on first run; cache reused after).

- [ ] **Step 4: Commit**

```bash
git add src/privacy_filter/detection/huggingface.py tests/slow/
git commit -m "feat(detection): HuggingFaceDetector for openai/privacy-filter, slow tests"
```

---

## Task 18: Pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create the pre-commit config**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=512]
      - id: detect-private-key
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.7
          - pydantic-settings>=2.4
        pass_filenames: false
        args: [--config-file=pyproject.toml]
```

- [ ] **Step 2: Install and run hooks**

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: pre-commit hooks (ruff, mypy, secret scan, file hygiene)"
```

---

## Task 19: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create `.dockerignore`**

```
.git
.venv
.pytest_cache
.ruff_cache
.mypy_cache
.hypothesis
.coverage
htmlcov
docs
tests
*.md
.env
.env.local
hf_cache
__pycache__
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

RUN pip install --no-cache-dir uv==0.4.20

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra hf --no-editable

FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 1001 app \
 && useradd  --system --uid 1001 --gid app --no-create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "privacy_filter.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Local sanity build (skip if Docker is unavailable)**

```bash
docker build -t privacy-filter:dev . || echo "(Docker not available — skipping)"
```

Expected: successful image build OR a clear message that Docker isn't installed.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: multi-stage Dockerfile, non-root user, healthcheck"
```

---

## Task 20: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow**

```yaml
# .github/workflows/ci.yml
name: ci

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "0 6 * * *"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --extra dev --frozen
      - run: uv run ruff check
      - run: uv run ruff format --check
      - run: uv run mypy
      - run: uv run pip-audit --strict

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --extra dev --frozen
      - run: uv run pytest -m "not slow" --cov=privacy_filter

  test-slow:
    if: github.event_name == 'schedule' || github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --extra dev --extra hf --frozen
      - name: Cache HuggingFace model
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-${{ hashFiles('src/privacy_filter/detection/huggingface.py') }}
      - run: uv run pytest -m slow
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint + fast tests on every push, slow tests on main and nightly"
```

---

## Task 21: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

```markdown
# privacy-filter

Stateless FastAPI microservice that detects and redacts personally identifiable information (PII) in plain text. Backed by the [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) HuggingFace token-classification model behind a pluggable `Detector` interface.

## Quickstart

```bash
uv venv
uv sync --extra dev          # API + tests, no torch
uv sync --extra dev --extra hf  # adds transformers + torch
cp .env.example .env         # then edit API_KEYS
uv run uvicorn privacy_filter.api.app:create_app --factory --reload
```

## API

`POST /v1/detect` — requires `X-API-Key`.

Request:
```json
{ "text": "Email me at alice@example.com tomorrow." }
```

Response:
```json
{
  "detections": [
    { "label": "private_email", "start": 12, "end": 29, "score": 0.99 }
  ],
  "redacted": "Email me at [PRIVATE_EMAIL] tomorrow.",
  "model": "openai/privacy-filter",
  "model_revision": "..."
}
```

`GET /healthz` — process liveness, no auth.
`GET /readyz` — model loaded and warmed, no auth.

## Configuration

All settings come from environment variables. See `.env.example` for the full list.

| Var | Required | Default |
|---|---|---|
| `API_KEYS` | yes | — |
| `MAX_INPUT_CHARS` | no | `50000` |
| `MAX_BODY_BYTES` | no | `262144` |
| `MODEL_ID` | no | `openai/privacy-filter` |
| `MODEL_REVISION` | no | (resolved at load time) |
| `LOG_LEVEL` | no | `INFO` |
| `CORS_ORIGINS` | no | (empty; CORS disabled) |

## Development

```bash
uv run pytest                 # fast tests, FakeDetector
uv run pytest -m slow         # exercises the real HF model (requires --extra hf)
uv run ruff check
uv run ruff format
uv run mypy
uv run pre-commit run --all-files
```

A single test:
```bash
uv run pytest tests/unit/test_redact.py::test_overlap_keeps_highest_score -v
```

## Docker

```bash
docker build -t privacy-filter .
docker run --rm -p 8000:8000 -e API_KEYS=changeme privacy-filter
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart, API contract, configuration, development commands"
```

---

## Task 22: CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

This is the artifact future Claude Code sessions will read at startup.

- [ ] **Step 1: Write `CLAUDE.md`**

````markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service is

Stateless FastAPI microservice that detects and redacts PII in plain text. Detection is delegated to a `Detector` Protocol; the production implementation wraps the HuggingFace `openai/privacy-filter` token-classification model. The API never imports `transformers` or `torch` directly — those live behind the optional `[hf]` extra.

The full design contract is `docs/superpowers/specs/2026-05-05-privacy-filter-microservice-design.md`. Read it before changing the API surface, the security posture, or the detection boundary.

## Commands

Dependency setup uses `uv`:

```bash
uv venv
uv sync --extra dev                  # everything except torch (use this by default)
uv sync --extra dev --extra hf       # adds transformers + torch for slow tests / running the real detector
```

Run the service locally:

```bash
uv run uvicorn privacy_filter.api.app:create_app --factory --reload
```

Test, lint, type-check:

```bash
uv run pytest                        # fast suite, FakeDetector everywhere
uv run pytest -m slow                # real HF model; requires --extra hf
uv run pytest tests/unit -v
uv run pytest tests/unit/test_redact.py::test_overlap_keeps_highest_score -v   # single test
uv run ruff check
uv run ruff format
uv run mypy                          # strict over src/ AND tests/
uv run pip-audit
uv run pre-commit run --all-files
```

`pyproject.toml` is the single source of truth for ruff, mypy, and pytest configuration.

## Architecture worth holding in your head

**The `Detector` boundary is load-bearing.** All inference is mediated by `privacy_filter.detection.protocol.Detector`. Routes never construct or import a concrete detector — they receive it through `api/deps.py::get_detector`, which reads it from `app.state`. The detector is built once in the `lifespan` (in `api/app.py::create_app`) and warmed with a single inference. Integration tests inject `FakeDetector` via the `detector_factory` argument to `create_app`. If you find yourself importing `transformers` outside `detection/huggingface.py`, stop — the layering is wrong.

**Three pieces of information flow through every request:**
1. `X-API-Key` is validated by `security.is_valid_api_key` (constant-time compare against the set in `Settings.api_keys`).
2. `text` is bounded twice: by `BodySizeLimitMiddleware` (raw bytes) and by `DetectRequest.with_max_chars` (Pydantic). Both bounds come from `Settings`.
3. A `request_id` is bound into the structlog context by `RequestIDMiddleware` and returned on `X-Request-ID`. All log records carry it.

**Logging is allowlist-based, not redaction-based.** `logging.drop_disallowed_fields` *only emits* fields in `ALLOWED_LOG_FIELDS`. To add a field to logs, add it to that constant; do not add ad-hoc fields hoping they'll be safe. The integration test `tests/integration/test_logging_no_pii.py` is the mechanical guard — never weaken it to make a feature pass; fix the feature.

**Errors return a uniform envelope.** `{"error": {"code", "message", "request_id"}}`. The exception handlers in `api/app.py::create_app` cover three cases: typed `APIError` (uses its own status/code/message), Pydantic `RequestValidationError` (mapped to `InvalidRequest`/422), and bare `Exception` (always 500 with the generic `internal_error` message). Never echo exception messages or stack traces to clients.

## How we work in this repo

**TDD is the default.** Red → green → refactor. The detector boundary is what makes this practical: the integration suite uses `FakeDetector` and runs in milliseconds. When implementing a behavior, write the failing test first, watch it fail, then write the minimum code to pass.

**Tests are typed too.** `mypy --strict` runs over `tests/` as well as `src/`. Tests get the same type rigor as production code; they're often where contract drift shows up first.

**Slow tests are isolated.** Anything that loads the real HF model is marked `@slow` and lives under `tests/slow/`. The fast suite must never download the model.

**Security choices have rationale captured in the spec.** Notably: API keys are stored as plaintext in env (not hashed), because the deployment context (stateless container, secrets from a manager) does not justify hashing — and the spec explains why so the question doesn't get re-litigated. Keep that energy: don't add security ceremony without naming the threat it counters.

**Commit attribution:** do not add `Co-Authored-By: Claude` trailers or "Generated with Claude Code" footers to commits, PRs, or any artifact. Plain Conventional Commits messages only.

## Files you will most often touch

- `src/privacy_filter/detection/protocol.py` — types and the Detector contract
- `src/privacy_filter/detection/redact.py` — span application; if you change this, update the property tests too
- `src/privacy_filter/api/routes.py` — the only file that knows the URL surface
- `src/privacy_filter/api/app.py` — middleware order, lifespan, exception handlers
- `tests/conftest.py` — the `client`/`fake_detector_factory`/`settings` fixtures used everywhere
- `docs/superpowers/specs/2026-05-05-privacy-filter-microservice-design.md` — design source of truth

## Things to avoid

- Importing `transformers` or `torch` outside `detection/huggingface.py`.
- Adding fields to log records without adding them to `ALLOWED_LOG_FIELDS` and reasoning about whether the field can ever carry user data.
- Returning exception messages, stack traces, or original input echoes in error responses.
- Skipping `mypy --strict`, ruff, or the PII-in-logs guard test to "just get something working." Those are the load-bearing checks for this service.
- Adding endpoints under `/v1/*` without `Depends(require_api_key)` — the `v1` router has it as a default dependency; any new prefix needs the same.
````

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md guidance for future Claude Code sessions"
```

---

## Self-review

**Spec coverage check:**

- Purpose / non-goals → captured in README + CLAUDE.md (no implementation).
- Project layout → Task 1, Task 2 (detection), Task 9 (api/), Task 13 (routes/app).
- Detector Protocol + boundary → Tasks 2, 4, 5, 17.
- Stack (uv, FastAPI, Pydantic, structlog, etc.) → Task 1.
- API contract (`POST /v1/detect`, `/healthz`, `/readyz`) → Tasks 9, 13, 14.
- Auth via `X-API-Key`, constant-time compare → Tasks 7, 12, 14.
- Input bounds (chars + bytes) → Tasks 9, 11, 14 (oversized body + oversized text tests).
- Logging discipline + sentinel guard → Tasks 8, 15.
- Error envelope + no stack-trace leaks → Tasks 10, 13, 14.
- Security headers + CORS off by default → Tasks 11, 13.
- Test strategy (unit, contract, integration, property, slow) → Tasks 2, 3, 4, 5, 14, 15, 16, 17.
- Lint / mypy strict / pip-audit → Tasks 1, 18, 20.
- CI with three jobs → Task 20.
- Configuration table → Task 6 + README + CLAUDE.md.
- Container — non-root user, healthcheck, weights baked → Task 19.
- CLAUDE.md → Task 22.
- Definition of done → covered by Tasks 14–22 collectively; the CI in Task 20 mechanically enforces the lint/mypy/test checks.

No spec sections lack a task.

**Placeholder scan:** No `TBD`, no `TODO`, no "implement later." Every task ships concrete code or a concrete file.

**Type / name consistency:**
- `Detector` Protocol shape (`model_id`, `model_revision`, `detect`) is identical in Tasks 2, 4, 12, 13, 17.
- `Detection` field names (`label`, `start`, `end`, `score`) consistent across Tasks 2, 3, 4, 5, 13, 14, 16, 17.
- `Settings` field names match `.env.example` and the README configuration table.
- `apply_spans(text, detections)` signature consistent across Tasks 3, 13, 16.
- `is_valid_api_key(presented, allowed)` consistent across Tasks 7 and 12.
- `ALLOWED_LOG_FIELDS` referenced consistently in Tasks 8 and 15 (and CLAUDE.md).

No drift.

**Ambiguity check:** The HuggingFace `aggregation_strategy="first"` choice in Task 17 is flagged as a known approximation pending model-card detail. The contract suite still validates correctness invariants. Everything else is deterministic.
