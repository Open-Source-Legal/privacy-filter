from privacy_filter.detection.protocol import Detection, Label
from privacy_filter.detection.redact import apply_spans


def test_no_detections_returns_input_unchanged() -> None:
    assert apply_spans("hello", []) == "hello"


def test_single_span_replaced() -> None:
    text = "Email me at alice@example.com tomorrow."
    d = Detection(label=Label.PRIVATE_EMAIL, start=12, end=29, score=0.99)
    assert apply_spans(text, [d]) == "Email me at [PRIVATE_EMAIL] tomorrow."


def test_multiple_spans_applied_in_reverse_so_offsets_stay_valid() -> None:
    text = "alice@x.com / 555-1212"
    spans = [
        Detection(label=Label.PRIVATE_EMAIL, start=0, end=11, score=0.9),
        Detection(label=Label.PRIVATE_PHONE, start=14, end=22, score=0.9),
    ]
    assert apply_spans(text, spans) == "[PRIVATE_EMAIL] / [PRIVATE_PHONE]"


def test_overlap_keeps_highest_score() -> None:
    text = "aaaaa"
    spans = [
        Detection(label=Label.SECRET, start=0, end=4, score=0.5),
        Detection(label=Label.PRIVATE_EMAIL, start=2, end=5, score=0.9),
    ]
    assert apply_spans(text, spans) == "aa[PRIVATE_EMAIL]"


def test_overlap_tie_break_by_earliest_start() -> None:
    text = "aaaaa"
    spans = [
        Detection(label=Label.SECRET, start=2, end=4, score=0.7),
        Detection(label=Label.PRIVATE_EMAIL, start=0, end=3, score=0.7),
    ]
    assert apply_spans(text, spans) == "[PRIVATE_EMAIL]aa"


def test_overlap_tie_break_by_longest_when_score_and_start_equal() -> None:
    text = "aaaaa"
    spans = [
        Detection(label=Label.SECRET, start=0, end=2, score=0.7),
        Detection(label=Label.PRIVATE_EMAIL, start=0, end=4, score=0.7),
    ]
    assert apply_spans(text, spans) == "[PRIVATE_EMAIL]a"


def test_zero_length_span_is_inserted() -> None:
    text = "abc"
    d = Detection(label=Label.SECRET, start=1, end=1, score=0.5)
    assert apply_spans(text, [d]) == "a[SECRET]bc"


def test_emoji_offsets_use_python_str_indexing() -> None:
    text = "hi 😀 alice@x.com"
    d = Detection(label=Label.PRIVATE_EMAIL, start=5, end=16, score=0.9)
    assert apply_spans(text, [d]) == "hi 😀 [PRIVATE_EMAIL]"


def test_adjacent_non_overlapping_spans_both_applied() -> None:
    text = "abcdef"
    spans = [
        Detection(label=Label.SECRET, start=0, end=3, score=0.9),
        Detection(label=Label.PRIVATE_EMAIL, start=3, end=6, score=0.9),
    ]
    assert apply_spans(text, spans) == "[SECRET][PRIVATE_EMAIL]"
