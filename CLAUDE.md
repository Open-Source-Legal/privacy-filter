# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service is

Stateless FastAPI microservice that detects PII in plain text. Detection is delegated to a `Detector` Protocol; the production implementation wraps the HuggingFace `openai/privacy-filter` token-classification model. The API never imports `transformers` or `torch` directly — those live behind the optional `[hf]` extra and are imported lazily inside `HuggingFaceDetector.__init__`.

V1 is detection-only. The service does not produce a redacted string; callers consume the spans (`{entity_group, score, word, start, end}`) and decide what to do with them.

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
uv run pytest                                  # fast suite (FakeDetector everywhere)
uv run pytest -m slow                          # real HF model; requires --extra hf
uv run pytest tests/unit -v                    # one directory
uv run pytest tests/unit/test_bioes.py::test_BIE_run_yields_one_span_with_min_score -v  # one test
uv run ruff check
uv run ruff format
uv run mypy                                    # strict over src/ AND tests/
uv run pip-audit
uv run pre-commit run --all-files
```

`pyproject.toml` is the single source of truth for ruff, mypy, and pytest configuration. The pre-commit hooks call `uv run` for ruff and mypy so they always use the same tool versions as the local env (avoids drift between `mirrors-mypy`/`ruff-pre-commit` pins and the project lock).

## Architecture worth holding in your head

**The `Detector` boundary is load-bearing.** All inference is mediated by `privacy_filter.detection.protocol.Detector`. Routes never construct or import a concrete detector — they receive it through `api/deps.py::get_detector`, which reads it from `app.state`. The detector is built once in the `lifespan` (in `api/app.py::create_app`) and warmed with a single inference. Integration tests inject `FakeDetector` via the `detector_factory` argument to `create_app`. If you find yourself importing `transformers` outside `detection/huggingface.py`, stop — the layering is wrong.

**BIOES grouping is ours, not HF's.** `HuggingFaceDetector` calls the pipeline with `aggregation_strategy="none"` and runs the per-subword tags through `detection/bioes.py::group_bioes`. Stock HF aggregation strategies are BIO-aware, not BIOES-aware (they treat `E-X` like `I-X` and merge spans that should split). The grouper is pure data → data, with no transformers/torch dependency, and it lives outside `huggingface.py` so it stays unit-testable. Don't switch to `aggregation_strategy="simple"` for "simplicity" — it changes the contract.

**Three pieces of information flow through every request:**

1. `X-API-Key` is validated by `security.is_valid_api_key` (constant-time compare against the set in `Settings.api_keys`).
2. `text` is bounded twice: by `BodySizeLimitMiddleware` (raw bytes) and by `bounded_detect_request(max_chars=settings.max_input_chars)` re-validation in the route. Both bounds come from `Settings`.
3. A `request_id` is bound into the structlog context by `RequestIDMiddleware` and returned on `X-Request-ID`. All log records carry it.

**Middleware is pure ASGI, not `BaseHTTPMiddleware`.** The three middlewares in `api/middleware.py` (`RequestIDMiddleware`, `SecurityHeadersMiddleware`, `BodySizeLimitMiddleware`) implement `async def __call__(scope, receive, send)`. We deliberately avoid `BaseHTTPMiddleware` because of its known interactions with Starlette's `ServerErrorMiddleware` re-raise contract — under `httpx.ASGITransport(raise_app_exceptions=True)` (the default), it leaks raised exceptions to the test client even after the 500 envelope is rendered. There is also a `SuppressHandledExceptionMiddleware` installed via a `build_middleware_stack` override in `api/app.py` that wraps everything outside `ServerErrorMiddleware` and swallows exceptions only when a response has already started. If you change exception handling, keep this in mind.

**Logging is allowlist-based, not redaction-based.** `logging.drop_disallowed_fields` *only emits* fields in `ALLOWED_LOG_FIELDS`. To add a field to logs, add it to that constant; do not add ad-hoc fields hoping they'll be safe. The integration test `tests/integration/test_logging_no_pii.py` is the mechanical guard — it deliberately reconfigures structlog *without* `drop_disallowed_fields`, so it proves call-sites never pass PII into log events, not just that the allowlist scrubs leaks. Never weaken it to make a feature pass; fix the feature.

**Errors return a uniform envelope.** `{"error": {"code", "message", "request_id"}}`. Three exception handlers in `api/app.py::create_app` cover: typed `APIError` (uses its own status/code/message), Pydantic `RequestValidationError` (mapped to `InvalidRequest`/422), and bare `Exception` (always 500 with the generic `internal_error` message). The route also catches `pydantic.ValidationError` from the bounded re-validation and re-raises as `InvalidRequest()`. Never echo exception messages or stack traces to clients.

## How we work in this repo

**TDD is the default.** Red → green → refactor. The detector boundary is what makes this practical: the integration suite uses `FakeDetector` and runs in milliseconds. When implementing a behavior, write the failing test first, watch it fail, then write the minimum code to pass.

**Tests are typed too.** `mypy --strict` runs over `tests/` as well as `src/`. Test functions need `-> None` return annotations even though ruff's `ANN` rule is disabled for `tests/**` — mypy strict and ruff are independent gates.

**Slow tests are isolated.** Anything that loads the real HF model is marked `@slow` and lives under `tests/slow/`. Each module starts with `pytest.importorskip("transformers")` so the fast suite's collection step doesn't fail when torch is absent.

**Security choices have rationale captured in the spec.** Notably: API keys are stored as plaintext in env (not hashed), because the deployment context (stateless container, secrets from a manager) does not justify hashing — and the spec explains why so the question doesn't get re-litigated. Keep that energy: don't add security ceremony without naming the threat it counters.

**Commit attribution:** do not add `Co-Authored-By: Claude` trailers or "Generated with Claude Code" footers to commits, PRs, comments, or any artifact. Plain Conventional Commits messages only.

## Files you will most often touch

- `src/privacy_filter/detection/protocol.py` — types and the Detector contract
- `src/privacy_filter/detection/bioes.py` — BIOES grouping; if you change this, run the unit tests AND the slow tests
- `src/privacy_filter/api/routes.py` — the only file that knows the URL surface
- `src/privacy_filter/api/app.py` — middleware order, lifespan, exception handlers, the build_middleware_stack override
- `src/privacy_filter/api/middleware.py` — pure-ASGI middlewares; preserve that pattern
- `tests/conftest.py` — the `client`/`fake_detector_factory`/`settings` fixtures used everywhere
- `docs/superpowers/specs/2026-05-05-privacy-filter-microservice-design.md` — design source of truth

## Things to avoid

- Importing `transformers` or `torch` outside `detection/huggingface.py`.
- Returning a redacted string from `/v1/detect` or any other endpoint. The service is detection-only.
- Switching `HuggingFaceDetector` to `aggregation_strategy="simple"`/`"first"` for "simplicity" — it changes the BIOES semantics.
- Converting the middlewares back to `BaseHTTPMiddleware` for "ergonomics" — see the exception-propagation note above.
- Adding fields to log records without adding them to `ALLOWED_LOG_FIELDS` and reasoning about whether the field can ever carry user data.
- Returning exception messages, stack traces, or original input echoes in error responses.
- Skipping `mypy --strict`, ruff, or the PII-in-logs guard test to "just get something working." Those are the load-bearing checks for this service.
- Adding endpoints under `/v1/*` without `Depends(require_api_key)` — the `v1` router has it as a default dependency; any new prefix needs the same guard.
- Adding any "Co-Authored-By: Claude" or "Generated with Claude Code" attribution to commits, PRs, or artifacts.
