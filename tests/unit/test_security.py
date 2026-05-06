import pytest

from privacy_filter.security import is_valid_api_key


def test_valid_key_accepted() -> None:
    assert is_valid_api_key("alpha", frozenset({"alpha", "beta"}))


def test_unknown_key_rejected() -> None:
    assert not is_valid_api_key("gamma", frozenset({"alpha", "beta"}))


def test_empty_key_rejected_even_if_set_contains_empty_string() -> None:
    # Defense in depth: never accept an empty presented key.
    assert not is_valid_api_key("", frozenset({"", "alpha"}))


def test_none_key_rejected() -> None:
    assert not is_valid_api_key(None, frozenset({"alpha"}))


def test_no_short_circuit_on_length(monkeypatch: pytest.MonkeyPatch) -> None:
    # Behavioral sanity: every allowed key must be compared, regardless of
    # the presented key's length or position. This catches an early-return bug.
    import privacy_filter.security as sec

    calls: list[tuple[str, str]] = []

    def fake_compare(a: str, b: str) -> bool:
        calls.append((a, b))
        return a == b

    monkeypatch.setattr(sec, "_compare", fake_compare)
    is_valid_api_key("xx", frozenset({"alpha", "beta", "gamma"}))
    assert len(calls) == 3
