# Privacy Filter Microservice — Design

**Date:** 2026-05-05
**Status:** Approved (revised 2026-05-05 — redaction removed from V1 scope)
**Owner:** scrudato@umich.edu

## Purpose

A FastAPI microservice that detects personally identifiable information (PII) in plain-text input and returns the structured detection spans. The service is stateless, containerized, and designed for high-throughput on-premises deployment behind an internal API gateway. The detection backend is the HuggingFace token-classification model `openai/privacy-filter`, accessed through a pluggable `Detector` interface so the model can be swapped without API changes.

V1 scope is intentionally narrow: plain text in, structured detection spans out. The service does **not** produce a redacted string — callers act on the spans however they need to (mask, replace, route, audit). File ingestion, async job queues, multi-tenant key management, and rate limiting are explicitly out of scope and will be designed separately if needed.

## Non-goals (V1)

- **Redaction / masking of input text.** The service returns spans; the caller decides how to use them.
- File upload or document parsing (PDF, DOCX, etc.)
- Asynchronous job submission / polling
- Per-tenant quotas or rate limiting
- A web UI or admin console
- Streaming responses
- Per-label confidence thresholds tuned for downstream tasks

## Architecture

### Project layout

```
src/privacy_filter/
  api/
    app.py          # FastAPI app factory, lifespan, exception handlers
    routes.py       # /v1/detect, /healthz, /readyz
    schemas.py      # Pydantic v2 request/response models
    deps.py         # dependency providers (auth, detector)
    errors.py       # custom exception types and JSON envelope mapper
    middleware.py   # request-id, security headers, body-size guard
  detection/
    protocol.py     # Detector Protocol, Detection dataclass, label enum
    huggingface.py  # HuggingFaceDetector implementation (optional extra)
    fake.py         # FakeDetector for tests
  config.py         # pydantic-settings, env-driven
  logging.py        # structlog setup, PII-safe processors
  security.py       # constant-time API-key comparison
tests/
  unit/
  integration/
  contract/         # Detector Protocol contract suite
  conftest.py
```

### Boundary: the `Detector` Protocol

All inference is mediated by a single Protocol so the API layer never imports `transformers`:

```python
class Detector(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def model_revision(self) -> str: ...
    def detect(self, text: str) -> list[Detection]: ...
```

`Detection` carries `(entity_group: Label, start: int, end: int, score: float, word: str)`:

- `entity_group` is one of 8 categories: `account_number`, `private_address`, `private_email`, `private_person`, `private_phone`, `private_url`, `private_date`, `secret`. (Internally a `StrEnum` named `Label`; over the wire it's a plain string.)
- `start` / `end` are character offsets in the original input string (Python slice semantics).
- `score` is the per-span confidence reported by the BIOES grouper (see below).
- `word` is the literal substring `text[start:end]` — included so callers can audit detections without re-tokenizing.

The `HuggingFaceDetector` runs the `transformers` token-classification pipeline for `openai/privacy-filter` with `aggregation_strategy="none"` to get per-subword tags, then applies a small **BIOES grouper** that collapses contiguous `B-X / I-X / E-X` runs (and standalone `S-X`) into spans, takes the minimum score across the run, maps offsets via the tokenizer's `offset_mapping`, and emits one `Detection` per group. Stock HF aggregation strategies are BIO-aware, not BIOES-aware, so we own the grouping rather than relying on `aggregation_strategy="simple"`/`"first"`. It exposes `model_revision` as the resolved HuggingFace commit SHA so callers can detect drift.

The detector is constructed once during the FastAPI `lifespan` and held on `app.state`. Routes receive it through a dependency provider, which integration tests override with `FakeDetector`.

### Stack

- Python 3.12
- `uv` for dependency management, lockfile, and virtual environments
- FastAPI + Uvicorn (ASGI)
- Pydantic v2 (request/response models) and `pydantic-settings` (config)
- `structlog` for logging
- `transformers` + `torch` for the HuggingFace detector — installed via the optional extra `[hf]`. Test runs do not install these; they use `FakeDetector`.
- `pytest`, `pytest-asyncio`, `httpx` (ASGI transport), `hypothesis` for property-based tests
- `ruff` (lint + format), `mypy --strict`, `pip-audit`, `pre-commit`

## API contract

### `POST /v1/detect`

**Auth:** `X-API-Key` header required. Missing or invalid → `401`.

**Request:**

```json
{ "text": "string" }
```

- `text` is required, non-empty, length `1..MAX_INPUT_CHARS` (default 50,000).
- Oversized payloads are rejected with `413` at the middleware layer before tokenization.

**Response 200:**

```json
{
  "detections": [
    {
      "entity_group": "private_email",
      "score": 0.98,
      "word": "alice@example.com",
      "start": 12,
      "end": 31
    }
  ],
  "model": "openai/privacy-filter",
  "model_revision": "<resolved HF commit SHA>"
}
```

- `entity_group` is one of the 8 model labels, returned verbatim (matches HF pipeline naming).
- `score` is the minimum per-token softmax score across the BIOES group (conservative aggregation).
- `word` is `text[start:end]` — the literal matched substring.
- `start` / `end` are character offsets in the original `text` (Python slice semantics: `text[start:end] == word`).
- The service does not transform the input text. Spans are returned as-is from the detector. Callers that need overlap resolution or masking implement that themselves.

### `GET /healthz`

No auth. Returns `200 {"status": "ok"}` whenever the process is responsive.

### `GET /readyz`

No auth. Returns `200 {"status": "ready"}` only after the detector has loaded and a single warm-up inference has completed during `lifespan`. Returns `503` otherwise.

### Error envelope

All non-2xx responses use a uniform shape:

```json
{ "error": { "code": "string", "message": "string", "request_id": "uuid" } }
```

Validation errors (`422`) map to this envelope; field paths are included but field *values* are not echoed back, to avoid surfacing PII in error responses or logs of those responses. Stack traces are never exposed; they go to logs only.

## Security posture

- **Authentication:** `X-API-Key` compared in constant time against the set of keys in env var `API_KEYS` (comma-separated). No hashing — the deployment context (stateless container, secrets injected from a manager) does not justify the added complexity. Rationale captured in this spec to prevent re-litigation: an env-dump attacker has plaintext either way.
- **Input bounds:** `MAX_INPUT_CHARS` enforced by Pydantic. ASGI body-size limit enforced in middleware. Both are configurable but default to safe values (50,000 chars / 256 KiB).
- **Logging:** `structlog`. The PII-safe processor enforces an allowlist of fields per record: `request_id`, `endpoint`, `method`, `status`, `latency_ms`, `input_chars`, `detection_count`, plus error metadata (`code`, exception *class* name — never the exception message, which could contain user data). Any field not on the allowlist is dropped before emission. Raw input, individual detections, query strings, and inbound headers are never logged. A test asserts that a sentinel input string never appears in any log record produced during a request.
- **Error handling:** Single exception handler converts all uncaught exceptions to the error envelope with `code="internal_error"` and a generic message. Detail (including the exception traceback) is logged with the request_id, never returned.
- **Headers:** Middleware sets `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`. HSTS is delegated to the ingress.
- **CORS:** Disabled by default. Enabled only via explicit env-driven origin allowlist.
- **Secrets:** Never read from files in the repo. `.env.example` ships placeholders only. `.gitignore` excludes `.env`. Pre-commit runs a secret-scan hook.
- **Dependencies:** `uv lock` is checked in. CI runs `pip-audit` against the lockfile and fails on advisories. Renovate or equivalent will be added later for automated bumps.

## Test strategy

TDD is the default workflow: red → green → refactor. The detector boundary keeps the suite fast — the bulk of tests run without installing torch.

**Pyramid:**

1. **Unit, pure logic.** `detection/protocol.py` validators (`start <= end`, `score in [0, 1]`, `entity_group` is a known enum member, `word == text[start:end]` invariant is the caller's responsibility). `detection/bioes.py` BIOES grouper (collapse `B/I/E`, standalone `S`, ignore `O`, take min score; verify on realistic tag sequences). `security.py` constant-time compare (correctness; doesn't short-circuit on length; rejects empty / missing / wrong).

2. **Contract suite over `Detector` Protocol.** A parameterized test module that any `Detector` implementation must pass: returns valid spans, offsets index real characters of the input (`text[d.start:d.end] == d.word`), `entity_group` comes from the closed set, idempotent on identical input. Always runs against `FakeDetector`; runs against `HuggingFaceDetector` only when `-m slow` is enabled.

3. **Integration, FastAPI app.** `httpx.AsyncClient` over ASGI transport with the `Detector` dependency overridden to `FakeDetector`:
   - 200 happy path with detection shape assertions (`entity_group`, `word`, `start`, `end`, `score`)
   - 401 on missing / wrong API key
   - 413 on oversized body, 422 on malformed JSON / wrong types / empty text
   - `/healthz` always 200; `/readyz` is 503 before lifespan completes, 200 after
   - **PII-in-logs guard:** request body contains a known sentinel string; the captured log output is asserted not to contain it. This test is the mechanical enforcement of the secure-logging rule.
   - Error envelope shape; no stack traces leak in 500 responses (induce by injecting a detector that raises).

4. **Slow tests** (`-m slow`, off by default). Loads `openai/privacy-filter` once per session and exercises a handful of golden inputs covering each of the 8 labels. Run on `main` and on a nightly schedule. The HF model cache is keyed in CI for reuse.

**Tooling discipline:**

- `ruff check` and `ruff format --check` clean
- `mypy --strict` clean over `src/` *and* `tests/`
- `pip-audit` clean against the committed lockfile
- pre-commit: ruff, mypy, secret scan, end-of-file fixer

## CI

GitHub Actions, single workflow, three jobs (Python 3.12 only):

1. **lint** — `uv sync --no-install-project --extra dev`, then `ruff check`, `ruff format --check`, `mypy --strict`, `pip-audit`. No torch install. Fast.
2. **test** — same install profile, runs `pytest -m 'not slow'`. Uses `FakeDetector` everywhere; never downloads the HF model.
3. **test-slow** — runs on pushes to `main` and on a nightly schedule. Installs the `[hf]` extra, runs `pytest -m slow` with the HF cache restored from a keyed action cache.

Caches: `uv` cache keyed on `uv.lock`; HF model cache keyed on the model revision pinned in code.

## Configuration

All configuration is environment-driven via `pydantic-settings`:

| Var | Default | Purpose |
|---|---|---|
| `API_KEYS` | (required) | Comma-separated valid API keys |
| `MAX_INPUT_CHARS` | `50000` | Pydantic validation bound on `text` |
| `MAX_BODY_BYTES` | `262144` | Middleware body-size guard |
| `MODEL_ID` | `openai/privacy-filter` | HF model name |
| `MODEL_REVISION` | (pinned SHA in code) | Forces a specific commit |
| `LOG_LEVEL` | `INFO` | structlog level |
| `CORS_ORIGINS` | (empty) | Allowlist; empty disables CORS |

`.env.example` ships with placeholders for every var. Real `.env` files are gitignored.

## Container

Single Dockerfile, multi-stage:

1. `uv` base, install `[hf]` extras into a venv
2. Copy `src/`, expose `8000`, run `uvicorn privacy_filter.api.app:create_app --factory`

Image runs as a non-root user. `HEALTHCHECK` calls `/healthz`. Model weights are baked into the image at build time (using `MODEL_REVISION`) so a cold start does not depend on outbound network.

## Open questions / explicit deferrals

- **Model details (architecture flags, dtype, device map).** User will provide; the `HuggingFaceDetector` constructor will accept these as kwargs sourced from settings.
- **Rate limiting.** Not in V1. If added later, prefer ingress-level (Envoy / nginx) over in-app middleware to keep the service stateless.
- **Observability.** No `/metrics` in V1. Structured JSON logs are sufficient until a real need appears.
- **Multi-language input.** The model's behavior on non-English text is the model's behavior; the service does not language-detect.

## Definition of done (V1)

- All endpoints implemented and covered by integration tests using `FakeDetector`
- `HuggingFaceDetector` passes the contract suite under `-m slow`
- BIOES grouper unit-tested against representative tag sequences
- ruff, mypy --strict, pip-audit all clean in CI
- PII-in-logs guard test passing
- Dockerfile builds and the container passes `/healthz` and `/readyz`
- `CLAUDE.md` documents commands and architecture for future Claude Code sessions
- `README.md` documents human-facing setup and usage
