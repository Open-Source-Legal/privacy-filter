from __future__ import annotations

from .protocol import Detection


def apply_spans(text: str, detections: list[Detection]) -> str:
    """Return text with each detected span replaced by `[<LABEL_UPPER>]`.

    Overlaps are resolved by keeping the highest-scoring span; ties are broken
    by earliest start, then longest length. The remaining non-overlapping spans
    are then applied in reverse offset order so earlier offsets stay valid.
    """
    if not detections:
        return text

    resolved = _resolve_overlaps(detections)
    out = text
    for d in sorted(resolved, key=lambda x: x.start, reverse=True):
        out = f"{out[: d.start]}[{d.label.value.upper()}]{out[d.end :]}"
    return out


def _resolve_overlaps(detections: list[Detection]) -> list[Detection]:
    ordered = sorted(
        detections,
        key=lambda d: (-d.score, d.start, -(d.end - d.start)),
    )
    kept: list[Detection] = []
    for d in ordered:
        if any(_overlaps(d, k) for k in kept):
            continue
        kept.append(d)
    return kept


def _overlaps(a: Detection, b: Detection) -> bool:
    return a.start < b.end and b.start < a.end
