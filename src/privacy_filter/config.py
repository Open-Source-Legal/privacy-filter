from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    api_keys: Annotated[frozenset[str], NoDecode]
    max_input_chars: Annotated[int, Field(ge=1)] = 50_000
    max_body_bytes: Annotated[int, Field(ge=1)] = 262_144
    model_id: str = "openai/privacy-filter"
    model_revision: str | None = None
    log_level: str = "INFO"
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ()

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
