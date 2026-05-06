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
    SECRET = "secret"  # noqa: S105 - label name, not a password


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
