from privacy_filter.detection.bioes import TaggedToken, group_bioes
from privacy_filter.detection.protocol import Label


def test_empty_token_stream_returns_no_detections() -> None:
    assert group_bioes("hello", []) == []


def test_all_outside_returns_no_detections() -> None:
    tokens = [
        TaggedToken(tag="O", score=0.9, start=0, end=5),
        TaggedToken(tag="O", score=0.9, start=5, end=11),
    ]
    assert group_bioes("hello world", tokens) == []


def test_singleton_S_tag_emits_one_detection() -> None:
    text = "send to alice"
    tokens = [
        TaggedToken(tag="O", score=0.99, start=0, end=4),
        TaggedToken(tag="O", score=0.99, start=5, end=7),
        TaggedToken(tag="S-private_person", score=0.95, start=8, end=13),
    ]
    out = group_bioes(text, tokens)
    assert len(out) == 1
    d = out[0]
    assert d.entity_group is Label.PRIVATE_PERSON
    assert d.start == 8
    assert d.end == 13
    assert d.score == 0.95
    assert d.word == "alice"


def test_BIE_run_yields_one_span_with_min_score() -> None:
    text = "alice@example.com"
    tokens = [
        TaggedToken(tag="B-private_email", score=0.95, start=0, end=5),
        TaggedToken(tag="I-private_email", score=0.80, start=5, end=14),
        TaggedToken(tag="E-private_email", score=0.90, start=14, end=17),
    ]
    out = group_bioes(text, tokens)
    assert len(out) == 1
    d = out[0]
    assert d.entity_group is Label.PRIVATE_EMAIL
    assert d.start == 0
    assert d.end == 17
    assert d.word == "alice@example.com"
    assert d.score == 0.80


def test_two_adjacent_spans_separated_by_O() -> None:
    text = "x y"
    tokens = [
        TaggedToken(tag="S-secret", score=0.9, start=0, end=1),
        TaggedToken(tag="O", score=0.99, start=1, end=2),
        TaggedToken(tag="S-private_email", score=0.85, start=2, end=3),
    ]
    out = group_bioes(text, tokens)
    assert [d.entity_group for d in out] == [Label.SECRET, Label.PRIVATE_EMAIL]


def test_label_switch_without_O_starts_new_span() -> None:
    text = "abc"
    tokens = [
        TaggedToken(tag="B-private_email", score=0.9, start=0, end=1),
        TaggedToken(tag="B-secret", score=0.9, start=1, end=2),
        TaggedToken(tag="E-secret", score=0.9, start=2, end=3),
    ]
    out = group_bioes(text, tokens)
    assert [d.entity_group for d in out] == [Label.PRIVATE_EMAIL, Label.SECRET]
    assert [d.start for d in out] == [0, 1]
    assert [d.end for d in out] == [1, 3]


def test_stray_I_without_B_is_treated_as_B() -> None:
    text = "abc"
    tokens = [
        TaggedToken(tag="I-private_email", score=0.7, start=0, end=2),
        TaggedToken(tag="E-private_email", score=0.9, start=2, end=3),
    ]
    out = group_bioes(text, tokens)
    assert len(out) == 1
    assert out[0].entity_group is Label.PRIVATE_EMAIL
    assert out[0].start == 0
    assert out[0].end == 3


def test_unknown_label_string_is_dropped() -> None:
    tokens = [
        TaggedToken(tag="B-bogus_label", score=0.9, start=0, end=3),
        TaggedToken(tag="E-bogus_label", score=0.9, start=3, end=5),
    ]
    assert group_bioes("xxxxx", tokens) == []


def test_unterminated_run_is_flushed_at_end() -> None:
    # Stream ends without an E- or O - still emit the in-progress span.
    text = "abcde"
    tokens = [
        TaggedToken(tag="B-secret", score=0.8, start=0, end=2),
        TaggedToken(tag="I-secret", score=0.7, start=2, end=5),
    ]
    out = group_bioes(text, tokens)
    assert len(out) == 1
    assert out[0].start == 0
    assert out[0].end == 5
    assert out[0].score == 0.7
