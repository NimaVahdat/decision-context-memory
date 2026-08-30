"""Text comparison utilities."""

from __future__ import annotations


def word_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity of word sets.

    Symmetric, bounded [0,1]. Appropriate for comparing
    short texts of similar length (rule vs rule).
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
