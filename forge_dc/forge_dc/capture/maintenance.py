"""Rulebook maintenance: dedup, supersession, preference detection.

Runs at consolidation time after rules are compiled.
No LLM calls — uses embedding similarity + word overlap + actor_id.

Classification matrix:
                    Same actor?
                   YES        NO
                ┌─────────┬──────────┐
Same text?  YES │ DUPLICATE│ AGREEMENT│
(high word      │ (merge)  │ (add     │
overlap)        │          │ supporter│
                ├─────────┼──────────┤
            NO  │SUPERSEDE │PREFERENCE│
(different      │(keep new │ CONFLICT │
content)        │remove old│(tag pair)│
                └─────────┴──────────┘
"""

from __future__ import annotations

import logging

from forge_dc.capture.models import (
    CompiledRule,
    RuleSource,
)
from forge_dc.shared.search import cosine_similarity
from forge_dc.shared.text import word_overlap

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TEXT_OVERLAP = 0.70
# Fallback cosine-similarity threshold for same-topic rule grouping,
# used when there aren't enough pairs to derive an adaptive threshold.
DEFAULT_TOPIC_SIM = 0.80
MIN_PAIRS_FOR_ADAPTIVE = 5

# Multi-signal same-topic detection weights
_W_EMBEDDING = 0.4
_W_KEYWORD = 0.3
_W_STRUCTURAL = 0.3
_TOPIC_THRESHOLD = 0.45  # on the combined score [0, 1]

# Common imperative verbs that start behavioral rules
_RULE_VERBS = {
    "format",
    "use",
    "include",
    "add",
    "show",
    "display",
    "omit",
    "remove",
    "exclude",
    "start",
    "end",
    "keep",
    "always",
    "never",
    "do",
    "apply",
    "present",
    "write",
    "list",
    "bold",
    "capitalize",
    "sort",
    "group",
    "tag",
}


def _find_adaptive_threshold(
    values: list[float],
    default: float,
) -> float:
    """Find threshold just below the top cluster of values.

    Finds all significant gaps (>0.1) in the sorted values.
    Picks the HIGHEST one (closest to 1.0) — this is the
    entry point to the "same topic" cluster. Threshold is
    just below that cluster's start.

    This avoids the midpoint problem where one outlier
    drags the threshold to useless territory.

    With < MIN_PAIRS_FOR_ADAPTIVE values, returns default.
    """
    if len(values) < MIN_PAIRS_FOR_ADAPTIVE:
        return default

    sorted_vals = sorted(values)

    # Find all significant gaps (>0.1)
    significant_gaps: list[tuple[int, float]] = []
    for i in range(len(sorted_vals) - 1):
        gap = sorted_vals[i + 1] - sorted_vals[i]
        if gap > 0.1:
            significant_gaps.append((i, gap))

    if not significant_gaps:
        return default

    # Pick the BIGGEST gap — it separates the main clusters.
    # (Not the highest — that picks small internal gaps
    # within the top cluster.)
    best_idx = max(
        significant_gaps,
        key=lambda x: x[1],
    )[0]

    # Threshold = just below the top cluster
    # Floor at 0.60 — "same general domain" (0.4-0.5) is NOT
    # "same specific topic." Only truly similar rules (0.7+)
    # should be considered same topic for supersession.
    threshold = max(
        sorted_vals[best_idx + 1] - 0.02,
        0.60,
    )

    logger.debug(
        "Adaptive threshold: %.3f (gap at %.3f→%.3f)",
        threshold,
        sorted_vals[best_idx],
        sorted_vals[best_idx + 1],
    )
    return threshold


def _compute_thresholds(
    all_rules: list[CompiledRule],
) -> tuple[float, float]:
    """Compute adaptive thresholds from the rule distribution.

    Returns (topic_threshold, text_threshold).
    """
    # Collect all pairwise embedding similarities
    emb_sims: list[float] = []
    text_overlaps: list[float] = []

    for i in range(len(all_rules)):
        emb_i = getattr(all_rules[i], "_embedding", None)
        for j in range(i + 1, len(all_rules)):
            emb_j = getattr(all_rules[j], "_embedding", None)
            if emb_i and emb_j:
                emb_sims.append(
                    cosine_similarity(emb_i, emb_j),
                )
            text_overlaps.append(
                word_overlap(
                    all_rules[i].text,
                    all_rules[j].text,
                ),
            )

    topic_thresh = _find_adaptive_threshold(
        emb_sims,
        DEFAULT_TOPIC_SIM,
    )
    text_thresh = _find_adaptive_threshold(
        text_overlaps,
        DEFAULT_TEXT_OVERLAP,
    )

    return topic_thresh, text_thresh


def maintain_rulebook(
    new_rules: list[CompiledRule],
    existing_rules: list[CompiledRule],
) -> list[CompiledRule]:
    """Maintain a clean rulebook by classifying new rules.

    For each new rule, compares against existing rules and:
    - DUPLICATE: skip new, existing is enough
    - AGREEMENT: add new actor to existing supporters
    - SUPERSESSION: remove old, keep new
    - PREFERENCE CONFLICT: tag both with conflict_group
    - UNRELATED: keep new as-is

    Returns the final list of rules to store (existing + new,
    after merges/removals).
    """
    # Work on a mutable copy
    rulebook = list(existing_rules)

    text_thresh = DEFAULT_TEXT_OVERLAP
    logger.debug(
        "Maintenance: %d new rules, %d existing, topic_threshold=%.2f (multi-signal), text=%.2f",
        len(new_rules),
        len(existing_rules),
        _TOPIC_THRESHOLD,
        text_thresh,
    )

    for new_rule in new_rules:
        # Tool rules skip maintenance — no actor preferences
        if new_rule.source == RuleSource.TOOL:
            _init_supporters(new_rule)
            rulebook.append(new_rule)
            continue

        _init_supporters(new_rule)

        # Find same-topic candidates (multi-signal)
        candidates = _find_same_topic(new_rule, rulebook)

        if not candidates:
            # UNRELATED — just add
            rulebook.append(new_rule)
            continue

        # Classify against the best candidate
        handled = False
        for candidate, emb_sim in candidates:
            # Skip tool rules in comparison
            if candidate.source == RuleSource.TOOL:
                continue

            overlap = word_overlap(new_rule.text, candidate.text)
            same_actor = (
                new_rule.actor_id == candidate.actor_id
                if new_rule.actor_id and candidate.actor_id
                else False
            )
            same_text = overlap > text_thresh

            if same_text and same_actor:
                # DUPLICATE — skip new
                logger.debug(
                    "DUPLICATE: '%s' ≈ '%s' (overlap=%.2f)",
                    new_rule.text[:40],
                    candidate.text[:40],
                    overlap,
                )
                handled = True
                break

            if same_text and not same_actor:
                # AGREEMENT — add supporter
                if new_rule.actor_id and (new_rule.actor_id not in candidate.supported_by):
                    candidate.supported_by.append(
                        new_rule.actor_id,
                    )
                logger.debug(
                    "AGREEMENT: '%s' +supporter %s",
                    candidate.text[:40],
                    new_rule.actor_id,
                )
                handled = True
                break

            if not same_text and same_actor:
                # SUPERSESSION — remove old, keep new
                rulebook.remove(candidate)
                rulebook.append(new_rule)
                logger.debug(
                    "SUPERSEDE: '%s' replaces '%s'",
                    new_rule.text[:40],
                    candidate.text[:40],
                )
                handled = True
                break

            if not same_text and not same_actor:
                # Different text, different actor — just keep
                # both. Scoring at query time handles actor
                # preference via the +0.3 actor boost.
                rulebook.append(new_rule)
                logger.debug(
                    "CROSS-ACTOR: '%s' (%s) vs '%s' (%s) — keeping both, scoring resolves",
                    new_rule.text[:30],
                    new_rule.actor_id,
                    candidate.text[:30],
                    candidate.actor_id,
                )
                handled = True
                break

        if not handled:
            rulebook.append(new_rule)

    return rulebook


def _init_supporters(rule: CompiledRule) -> None:
    """Ensure rule has its own actor in supported_by."""
    if not rule.supported_by and rule.actor_id:
        rule.supported_by = [rule.actor_id]


def _topic_score(
    rule_a: CompiledRule,
    rule_b: CompiledRule,
) -> float:
    """Multi-signal same-topic score between two rules.

    Combines three signals:
    1. Embedding similarity (semantic paraphrase detection)
    2. Keyword Jaccard overlap (lexical match for short rules)
    3. Structural match (same leading verb + shared domain nouns)

    Returns score in [0, 1]. Higher = more likely same topic.
    """
    # Signal 1: embedding similarity [0, 1]
    emb_a = getattr(rule_a, "_embedding", None)
    emb_b = getattr(rule_b, "_embedding", None)
    emb_sim = 0.0
    if emb_a and emb_b:
        emb_sim = max(cosine_similarity(emb_a, emb_b), 0.0)

    # Signal 2: keyword Jaccard [0, 1]
    words_a = set(rule_a.text.lower().split())
    words_b = set(rule_b.text.lower().split())
    union = words_a | words_b
    kw_jaccard = len(words_a & words_b) / len(union) if union else 0.0

    # Signal 3: structural match [0, 1]
    # Based on shared DOMAIN NOUNS — the verb alone is not
    # enough because ADD/INCLUDE/USE appear in most rules.
    # Verb match is a small bonus, noun overlap is the signal.
    import re as _re

    stopwords = _RULE_VERBS | {
        "the",
        "a",
        "an",
        "in",
        "of",
        "for",
        "to",
        "and",
        "or",
        "not",
        "all",
        "with",
        "as",
        "when",
        "that",
        "this",
        "from",
        "into",
        "section",
        "available",
        "every",
        "always",
    }
    # Strip punctuation before matching
    clean_a = {_re.sub(r"[^\w]", "", w) for w in words_a}
    clean_b = {_re.sub(r"[^\w]", "", w) for w in words_b}
    nouns_a = {w for w in clean_a if w not in stopwords and len(w) > 3}
    nouns_b = {w for w in clean_b if w not in stopwords and len(w) > 3}
    noun_union = nouns_a | nouns_b

    structural = 0.0
    if noun_union:
        noun_overlap = len(nouns_a & nouns_b) / len(noun_union)
        structural = noun_overlap

        # Small bonus if verb also matches
        tokens_a = rule_a.text.lower().split()
        tokens_b = rule_b.text.lower().split()
        if tokens_a and tokens_b:
            verb_a = tokens_a[0].rstrip(":.")
            verb_b = tokens_b[0].rstrip(":.")
            if verb_a == verb_b and verb_a in _RULE_VERBS:
                structural = min(structural + 0.15, 1.0)

    score = _W_EMBEDDING * emb_sim + _W_KEYWORD * kw_jaccard + _W_STRUCTURAL * structural
    return score


def _find_same_topic(
    new_rule: CompiledRule,
    existing: list[CompiledRule],
    threshold: float = _TOPIC_THRESHOLD,
) -> list[tuple[CompiledRule, float]]:
    """Find existing rules about the same topic.

    Uses multi-signal scoring (embedding + keyword + structural).
    Returns candidates sorted by score descending.
    """
    candidates: list[tuple[CompiledRule, float]] = []
    for rule in existing:
        score = _topic_score(new_rule, rule)
        if score > threshold:
            candidates.append((rule, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates
