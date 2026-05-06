from dataclasses import FrozenInstanceError

import pytest

from privacy_filter.detection.protocol import Detection, Label


def test_label_values_match_model_card() -> None:
    expected = {
        "account_number",
        "private_address",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "private_date",
        "secret",
    }
    assert {member.value for member in Label} == expected


def test_detection_accepts_valid_fields() -> None:
    d = Detection(
        entity_group=Label.PRIVATE_EMAIL,
        start=0,
        end=10,
        score=0.9,
        word="alice@x.com",
    )
    assert d.entity_group is Label.PRIVATE_EMAIL
    assert d.start == 0
    assert d.end == 10
    assert d.score == 0.9
    assert d.word == "alice@x.com"


def test_detection_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start must be >= 0"):
        Detection(entity_group=Label.SECRET, start=-1, end=5, score=0.5, word="x")


def test_detection_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match=r"end .* must be >= start"):
        Detection(entity_group=Label.SECRET, start=10, end=5, score=0.5, word="")


def test_detection_allows_zero_length_span() -> None:
    Detection(entity_group=Label.SECRET, start=5, end=5, score=0.5, word="")


def test_detection_rejects_score_below_zero() -> None:
    with pytest.raises(ValueError, match="score must be in"):
        Detection(entity_group=Label.SECRET, start=0, end=1, score=-0.1, word="a")


def test_detection_rejects_score_above_one() -> None:
    with pytest.raises(ValueError, match="score must be in"):
        Detection(entity_group=Label.SECRET, start=0, end=1, score=1.1, word="a")


def test_detection_is_frozen() -> None:
    d = Detection(entity_group=Label.SECRET, start=0, end=1, score=0.5, word="a")
    with pytest.raises(FrozenInstanceError):
        d.start = 99  # type: ignore[misc]
