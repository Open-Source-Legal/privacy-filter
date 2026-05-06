import pytest

from privacy_filter.config import Settings


def test_settings_loads_required_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "key-a,key-b")
    s = Settings()
    assert s.api_keys == frozenset({"key-a", "key-b"})


def test_settings_rejects_empty_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "")
    with pytest.raises(ValueError, match="API_KEYS"):
        Settings()


def test_settings_strips_whitespace_and_drops_empties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEYS", " key-a , ,key-b ,")
    s = Settings()
    assert s.api_keys == frozenset({"key-a", "key-b"})


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "k")
    s = Settings()
    assert s.max_input_chars == 50_000
    assert s.max_body_bytes == 262_144
    assert s.model_id == "openai/privacy-filter"
    assert s.model_revision is None
    assert s.log_level == "INFO"
    assert s.cors_origins == ()


def test_settings_parses_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "k")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example,https://b.example")
    s = Settings()
    assert s.cors_origins == ("https://a.example", "https://b.example")
