from privacy_filter.detection.fake import FakeDetector
from privacy_filter.detection.protocol import Detection, Label
from tests.contract.detector_contract import assert_detector_contract


def test_fake_detector_satisfies_contract() -> None:
    def script(text: str) -> list[Detection]:
        if "@" in text:
            idx = text.index("@")
            start = max(0, idx - 5)
            end = min(len(text), idx + 5)
            return [
                Detection(
                    entity_group=Label.PRIVATE_EMAIL,
                    start=start,
                    end=end,
                    score=0.9,
                    word=text[start:end],
                )
            ]
        return []

    detector = FakeDetector(script, model_id="fake", model_revision="r1")

    assert_detector_contract(
        detector,
        inputs=[
            "hello world",
            "alice@example.com",
            "  alice@example.com",
            "no detections here",
            "",
        ],
    )
