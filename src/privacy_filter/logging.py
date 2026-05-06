from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
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
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
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
        cache_logger_on_first_use=False,
    )
