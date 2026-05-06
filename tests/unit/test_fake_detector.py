from privacy_filter.detection.fake import FakeDetector
from privacy_filter.detection.protocol import Detection, Label


def test_fake_detector_returns_scripted_detections() -> None:
    spans = [
        Detection(
            entity_group=Label.PRIVATE_EMAIL,
            start=0,
            end=5,
            score=0.9,
            word="hello",
        )
    ]
    detector = FakeDetector(spans)
    assert detector.detect("hello") == spans


def test_fake_detector_exposes_model_id_and_revision() -> None:
    detector = FakeDetector([], model_id="fake-model", model_revision="rev-1")
    assert detector.model_id == "fake-model"
    assert detector.model_revision == "rev-1"


def test_fake_detector_supports_callable_for_per_input_responses() -> None:
    def script(text: str) -> list[Detection]:
        if "alice" in text:
            return [
                Detection(
                    entity_group=Label.PRIVATE_PERSON,
                    start=0,
                    end=5,
                    score=1.0,
                    word=text[0:5],
                )
            ]
        return []

    detector = FakeDetector(script)
    out = detector.detect("alice")
    assert len(out) == 1
    assert out[0].entity_group is Label.PRIVATE_PERSON
    assert detector.detect("bob") == []


def test_fake_detector_default_metadata() -> None:
    detector = FakeDetector([])
    assert detector.model_id == "fake"
    assert detector.model_revision == "test"
