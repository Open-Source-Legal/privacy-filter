import pytest

# Skip the whole module if the [hf] extra isn't installed.
pytest.importorskip("transformers")
pytest.importorskip("torch")

slow = pytest.mark.slow


@slow
def test_huggingface_detector_satisfies_contract() -> None:
    from privacy_filter.detection.huggingface import HuggingFaceDetector
    from tests.contract.detector_contract import assert_detector_contract

    detector = HuggingFaceDetector(model_id="openai/privacy-filter")
    assert_detector_contract(
        detector,
        inputs=[
            "",
            "no pii here",
            "Email alice@example.com tomorrow.",
            "Call me at +1 415 555 0123 please.",
            "My SSN is 123-45-6789.",
        ],
    )


@slow
def test_huggingface_detector_finds_email() -> None:
    from privacy_filter.detection.huggingface import HuggingFaceDetector
    from privacy_filter.detection.protocol import Label

    detector = HuggingFaceDetector(model_id="openai/privacy-filter")
    out = detector.detect("Please email alice@example.com soon.")
    assert any(d.entity_group is Label.PRIVATE_EMAIL for d in out)
