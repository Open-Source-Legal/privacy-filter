from dataclasses import FrozenInstanceError

import pytest

from privacy_filter.detection.protocol import Detection, Label


def test_label_values_match_model_card() -> None:
    expected = {
        "account_number", "private_address", "private_email", "private_person",
        "private_phone", "private_url", "private_date", "secret",
    }
    assert {member.value for member in Label} == expected


def test_detection_accepts_valid_fields() -> None:
    d = Detection(label=Label.PRIVATE_EMAIL, start=0, end=10, score=0.9)
    assert d.label is Label.PRIVATE_EMAIL
    assert d.start == 0
    assert d.end == 10
    assert d.score == 0.9


def test_detection_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start must be >= 0"):
        Detection(label=Label.SECRET, start=-1, end=5, score=0.5)


def test_detection_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match=r"end .* must be >= start"):
        Detection(label=Label.SECRET, start=10, end=5, score=0.5)


def test_detection_allows_zero_length_span() -> None:
    Detection(label=Label.SECRET, start=5, end=5, score=0.5)


def test_detection_rejects_score_below_zero() -> None:
    with pytest.raises(ValueError, match="score must be in"):
        Detection(label=Label.SECRET, start=0, end=1, score=-0.1)


def test_detection_rejects_score_above_one() -> None:
    with pytest.raises(ValueError, match="score must be in"):
        Detection(label=Label.SECRET, start=0, end=1, score=1.1)


def test_detection_is_frozen() -> None:
    d = Detection(label=Label.SECRET, start=0, end=1, score=0.5)
    with pytest.raises(FrozenInstanceError):
        d.start = 99  # type: ignore[misc]
