from __future__ import annotations

import json
from typing import Any

import structlog

from privacy_filter.logging import (
    ALLOWED_LOG_FIELDS,
    configure_logging,
    drop_disallowed_fields,
)


def test_allowlist_drops_disallowed_keys() -> None:
    event: dict[str, Any] = {
        "event": "request_completed",
        "request_id": "abc",
        "endpoint": "/v1/detect",
        "status": 200,
        "latency_ms": 12,
        "input_chars": 42,
        "detection_count": 3,
        # disallowed:
        "text": "alice@example.com",
        "headers": {"X-API-Key": "secret"},
        "detections": [{"word": "alice@example.com"}],
    }
    cleaned = drop_disallowed_fields(None, "info", event)
    assert "text" not in cleaned
    assert "headers" not in cleaned
    assert "detections" not in cleaned
    assert cleaned["request_id"] == "abc"
    assert cleaned["status"] == 200
    assert cleaned["detection_count"] == 3


def test_allowed_fields_match_spec() -> None:
    must_contain = {
        "event", "request_id", "endpoint", "method", "status", "latency_ms",
        "input_chars", "detection_count", "code", "exc_class",
    }
    assert must_contain <= ALLOWED_LOG_FIELDS


def test_configure_logging_emits_structured_json(capsys: Any) -> None:
    configure_logging(level="INFO")
    log = structlog.get_logger("test")
    log.info("request_completed", request_id="abc", status=200, text="leak")

    output = capsys.readouterr().out.strip().splitlines()
    payloads = [json.loads(line) for line in output if line.strip().startswith("{")]
    assert payloads, "expected at least one JSON log line"
    record = payloads[-1]
    assert record["event"] == "request_completed"
    assert record["request_id"] == "abc"
    assert record["status"] == 200
    assert "text" not in record


def test_configure_logging_respects_level(capsys: Any) -> None:
    configure_logging(level="WARNING")
    log = structlog.get_logger("test")
    log.info("ignored", request_id="abc")
    log.warning("kept", request_id="abc")

    output = capsys.readouterr().out.strip().splitlines()
    payloads = [json.loads(line) for line in output if line.strip().startswith("{")]
    events = [p["event"] for p in payloads]
    assert "ignored" not in events
    assert "kept" in events
