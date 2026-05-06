"""Group per-subword BIOES token-classification predictions into spans.

Input: an ordered sequence of (tag_str, score, char_start, char_end) tuples
where ``tag_str`` is one of ``O``, ``B-<label>``, ``I-<label>``, ``E-<label>``,
``S-<label>``. Output: a list of ``Detection`` objects with the minimum
score across the run, char offsets covering the whole run, and ``word`` set
to the slice of the original text.

The grouper is BIOES-aware: stock HuggingFace ``aggregation_strategy``
options treat ``E-`` like ``I-`` and merge spans that should split, so we
own this logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .protocol import Detection, Label


@dataclass(frozen=True, slots=True)
class TaggedToken:
    tag: str  # "O", "B-private_email", "I-private_email", etc.
    score: float
    start: int
    end: int


def group_bioes(text: str, tokens: Iterable[TaggedToken]) -> list[Detection]:
    out: list[Detection] = []
    open_label: Label | None = None
    open_start: int = 0
    open_end: int = 0
    open_min_score: float = 1.0

    def flush() -> None:
        nonlocal open_label
        if open_label is None:
            return
        out.append(
            Detection(
                entity_group=open_label,
                start=open_start,
                end=open_end,
                score=open_min_score,
                word=text[open_start:open_end],
            )
        )
        open_label = None

    for tok in tokens:
        prefix, _, raw_label = tok.tag.partition("-")
        if prefix == "O" or not raw_label:
            flush()
            continue

        try:
            label = Label(raw_label)
        except ValueError:
            # Unknown label string - treat as background (defensive).
            flush()
            continue

        if prefix == "S":
            flush()
            out.append(
                Detection(
                    entity_group=label,
                    start=tok.start,
                    end=tok.end,
                    score=tok.score,
                    word=text[tok.start : tok.end],
                )
            )
            continue

        if prefix == "B":
            flush()
            open_label = label
            open_start = tok.start
            open_end = tok.end
            open_min_score = tok.score
            continue

        if prefix in ("I", "E"):
            if open_label != label:
                # Mid-sentence label switch or stray I/E without a B -
                # treat this token as a fresh B (defensive recovery).
                flush()
                open_label = label
                open_start = tok.start
                open_end = tok.end
                open_min_score = tok.score
            else:
                open_end = tok.end
                open_min_score = min(open_min_score, tok.score)

            if prefix == "E":
                flush()
            continue

        # Unknown prefix - be safe and flush.
        flush()

    flush()
    return out
