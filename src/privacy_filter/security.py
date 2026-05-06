from __future__ import annotations

import hmac


def _compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def is_valid_api_key(presented: str | None, allowed: frozenset[str]) -> bool:
    """Constant-time API-key check.

    Always evaluates ``_compare`` against every allowed key so timing behavior
    does not leak information about which key matched (or how close a near-miss
    was). Empty/missing presented keys are rejected up front — never accept
    an empty string even if the allowed set somehow contains one.
    """
    if not presented:
        return False
    matched = False
    for candidate in allowed:
        if _compare(presented, candidate):
            matched = True
    return matched
