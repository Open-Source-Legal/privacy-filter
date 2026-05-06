from __future__ import annotations

import io
import json
import logging

import pytest
import structlog
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_no_log_record_contains_input_text(client: AsyncClient, api_key: str) -> None:
    sentinel = "SECRET_SENTINEL_alice@uniq-test.example_4f9a"

    # Capture both stdlib logging and structlog stream output.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)

    # Re-route structlog to the same buffer for the duration of this test.
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
        assert sentinel not in captured, f"sentinel leaked into logs:\n{captured}"

        # Structured records remain valid JSON when present.
        for line in captured.splitlines():
            if line.strip().startswith("{"):
                json.loads(line)
    finally:
        root.removeHandler(handler)
