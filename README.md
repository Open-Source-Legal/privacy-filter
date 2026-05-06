# privacy-filter

Stateless FastAPI microservice that detects personally identifiable information (PII) in plain text and returns the structured detection spans. The detection backend is the HuggingFace token-classification model [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter), accessed through a pluggable `Detector` interface so the model can be swapped without API changes.

V1 is intentionally narrow: plain text in, detection spans out. No redaction, no file ingestion, no async job queue. Callers act on the spans however they need to.

## Quickstart

```bash
uv venv
uv sync --extra dev                 # API + tests, no torch
# Or, to run the real detector locally:
uv sync --extra dev --extra hf      # adds transformers + torch (multi-GB)

cp .env.example .env                # then edit API_KEYS
uv run uvicorn privacy_filter.api.app:create_app --factory --reload
```

## API

### `POST /v1/detect`

**Header:** `X-API-Key: <one of API_KEYS>` (required)

**Request body:**
```json
{ "text": "Email me at alice@example.com tomorrow." }
```

**Response 200:**
```json
{
  "detections": [
    {
      "entity_group": "private_email",
      "score": 0.99,
      "word": "alice@example.com",
      "start": 12,
      "end": 29
    }
  ],
  "model": "openai/privacy-filter",
  "model_revision": "<resolved HF commit SHA>"
}
```

`entity_group` is one of: `account_number`, `private_address`, `private_email`, `private_person`, `private_phone`, `private_url`, `private_date`, `secret`. `start`/`end` are character offsets in the original input (Python slice semantics: `text[start:end] == word`).

### `GET /healthz`

Liveness check. No auth. Returns `200 {"status": "ok"}` whenever the process is responsive.

### `GET /readyz`

Readiness check. No auth. Returns `200 {"status": "ready"}` only after the detector has loaded and warmed up. Returns `503` otherwise.

### Errors

All non-2xx responses use a uniform envelope:

```json
{ "error": { "code": "string", "message": "string", "request_id": "uuid" } }
```

Codes: `unauthorized`, `payload_too_large`, `invalid_request`, `not_ready`, `internal_error`. Error responses never include exception messages or stack traces.

## Configuration

All settings come from environment variables. See `.env.example`.

| Var | Required | Default |
|---|---|---|
| `API_KEYS` | yes | — |
| `MAX_INPUT_CHARS` | no | `50000` |
| `MAX_BODY_BYTES` | no | `262144` |
| `MODEL_ID` | no | `openai/privacy-filter` |
| `MODEL_REVISION` | no | (resolved at load time) |
| `LOG_LEVEL` | no | `INFO` |
| `CORS_ORIGINS` | no | (empty; CORS disabled) |

`API_KEYS` is comma-separated. CORS stays disabled unless `CORS_ORIGINS` is set.

## Development

```bash
uv run pytest                                  # fast tests (FakeDetector)
uv run pytest -m slow                          # exercises the real HF model (requires --extra hf)
uv run pytest tests/unit/test_bioes.py -v      # one test file
uv run pytest tests/unit/test_bioes.py::test_BIE_run_yields_one_span_with_min_score -v   # one test
uv run ruff check
uv run ruff format
uv run mypy
uv run pip-audit
uv run pre-commit run --all-files
```

The fast suite uses a `FakeDetector` injected via `create_app(detector_factory=...)`, so it runs in milliseconds and does not require `transformers` or `torch`. Slow tests live under `tests/slow/` and are gated by the `slow` pytest marker.

## Docker

```bash
docker build -t privacy-filter .
docker run --rm -p 8000:8000 -e API_KEYS=changeme privacy-filter
```

The image runs as a non-root user (`uid 1001`) and includes a `HEALTHCHECK` against `/healthz`. The first container start downloads the HF model weights into the user's HF cache; mount a volume at `/home/app/.cache/huggingface` (or set `HF_HOME`) to persist them across container instances. Pre-baking weights into the image is a planned hardening for air-gapped deployments — see the design spec.

## Architecture

The detector boundary is the central abstraction. All inference is mediated by `privacy_filter.detection.protocol.Detector`, a Protocol with `model_id`, `model_revision`, and `detect(text) -> list[Detection]`. Routes never construct or import a concrete detector — they receive it through `api/deps.py::get_detector`, which reads it from `app.state` after the `lifespan` populates it. Integration tests inject `FakeDetector`; production uses `HuggingFaceDetector` (lazy-imports torch).

Logging uses an allowlist processor: only operational metadata fields (`request_id`, `endpoint`, `method`, `status`, `latency_ms`, `input_chars`, `detection_count`, plus error metadata) are emitted; everything else is dropped before the JSON renderer sees it. The integration test `tests/integration/test_logging_no_pii.py` is the mechanical guard.

The full design contract is in `docs/superpowers/specs/2026-05-05-privacy-filter-microservice-design.md`.
