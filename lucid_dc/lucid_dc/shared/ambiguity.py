"""Deterministic ambiguity detection for corrections.

Decides whether a correction is self-contained (the rule
text alone is enough for the agent) or context-dependent
(the agent needs evidence/outcome to understand it).

Used at capture time alongside aftermath classification
to set correction_needs_context on the document.
"""

from __future__ import annotations

# Markers that suggest the correction references
# something from the conversation context rather than
# stating a self-contained rule.
_AMBIGUOUS_MARKERS = [
    "that ",
    "that,",
    "this ",
    "this,",
    " it ",
    " it,",
    " it.",
    "the same",
    "like before",
    "like last time",
    "remove it",
    "fix it",
    "change it",
    "the one ",
    "what you had",
    "previous ",
    "the last ",
    "that part",
    "that section",
    "that thing",
]


def correction_needs_context(correction_text: str) -> bool:
    """Deterministic check: is this correction ambiguous?

    Returns True if the correction likely depends on
    conversation context to be understood. Returns False
    if the correction is a self-contained rule.

    This is the fast path — the LLM fallback in
    classify_followup can override this when it fires.
    """
    if not correction_text:
        return False

    words = correction_text.split()

    # Very short corrections are likely ambiguous
    # ("fix that", "not like that")
    if len(words) < 5:
        return True

    lower = correction_text.lower()
    return any(m in lower for m in _AMBIGUOUS_MARKERS)
