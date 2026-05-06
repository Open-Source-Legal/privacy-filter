"""Reusable contract assertions for any Detector implementation.

A test module wires up its concrete detector and golden inputs, then calls
``assert_detector_contract``. This keeps the contract in one place so the
fast suite (FakeDetector) and the slow suite (HuggingFaceDetector) share it.
"""

from __future__ import annotations

from collections.abc import Iterable

from privacy_filter.detection.protocol import Detector, Label


def assert_detector_contract(
    detector: Detector,
    inputs: Iterable[str],
) -> None:
    materialized = list(inputs)

    assert isinstance(detector.model_id, str), "Detector.model_id must be a string"
    assert detector.model_id, "Detector.model_id must be non-empty"
    assert isinstance(detector.model_revision, str), "Detector.model_revision must be a string"

    label_values = {member.value for member in Label}

    for text in materialized:
        first = detector.detect(text)
        second = detector.detect(text)
        assert first == second, "detect must be deterministic for identical input"

        for d in first:
            assert d.entity_group.value in label_values, f"unknown entity_group: {d.entity_group}"
            assert 0 <= d.start <= d.end <= len(text), (
                f"span out of range: ({d.start}, {d.end}) for len {len(text)}"
            )
            assert 0.0 <= d.score <= 1.0, f"score out of range: {d.score}"
            assert d.word == text[d.start : d.end], (
                f"word mismatch: {d.word!r} != text[{d.start}:{d.end}]={text[d.start : d.end]!r}"
            )
