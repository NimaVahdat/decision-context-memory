"""Scoring functions for forge_dc rule retrieval.

Two-context weighted scoring (0.6 task + 0.4 data) for both
embedding and BM25. RRF for combining signals. Multiplicative
boosts for actor/subject.
"""

from __future__ import annotations

from forge_dc.capture.models import (
    CompiledRule,
)
from forge_dc.shared.search import bm25_score, cosine_similarity

RRF_K = 60


# ── Two-context scoring ─────────────────────────────────────


def weighted_emb_score(
    task_emb: list[float] | None,
    data_emb: list[float] | None,
    candidate_emb: list[float],
) -> float:
    """Weighted two-context embedding: 0.6 task + 0.4 data."""
    task_sim = cosine_similarity(task_emb, candidate_emb) if task_emb else 0.0
    data_sim = cosine_similarity(data_emb, candidate_emb) if data_emb else 0.0

    if task_emb and data_emb:
        return 0.6 * task_sim + 0.4 * data_sim
    return task_sim or data_sim


def weighted_bm25_score(
    task_text: str,
    data_text: str,
    candidate_text: str,
) -> float:
    """Weighted two-context BM25: 0.6 task + 0.4 data."""
    task_bm = bm25_score(task_text, candidate_text) if task_text else 0.0
    data_bm = bm25_score(data_text, candidate_text) if data_text else 0.0

    if task_text and data_text:
        return 0.6 * task_bm + 0.4 * data_bm
    return task_bm or data_bm


# ── RRF helpers ──────────────────────────────────────────────


def rrf_from_ranks(
    rule_id: str,
    *rank_maps: dict[str, int],
    k: int = RRF_K,
    n_fallback: int = 1000,
) -> float:
    """RRF score from multiple rank maps."""
    score = 0.0
    for rm in rank_maps:
        rank = rm.get(rule_id, n_fallback)
        score += 1.0 / (k + rank)
    return score


def build_rank_map(
    scored: list[tuple[str, float]],
) -> dict[str, int]:
    """Build {id: rank} from a sorted (id, score) list."""
    return {rid: i for i, (rid, _) in enumerate(scored)}


# ── PATH 2a: Correction rule scoring ────────────────────────


def score_correction_rules(
    rules: list[CompiledRule],
    task_context: str,
    data_context: str,
    task_emb: list[float] | None,
    data_emb: list[float] | None,
    actor_id: str | None = None,
    subjects: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Score correction rules: RRF(embedding, BM25) × boosts.

    Two-context weighted scoring. Multiplicative boosts for
    actor and subject overlap.
    """
    if not rules:
        return []

    # Raw signals
    emb_scores: list[tuple[str, float]] = []
    bm25_scores: list[tuple[str, float]] = []

    for rule in rules:
        rule_emb = getattr(rule, "_embedding", None)
        emb = 0.0
        if rule_emb and (task_emb or data_emb):
            emb = weighted_emb_score(
                task_emb,
                data_emb,
                rule_emb,
            )
        emb_scores.append((rule.id, emb))

        bm = weighted_bm25_score(
            task_context,
            data_context,
            rule.text,
        )
        bm25_scores.append((rule.id, bm))

    # Rank independently
    emb_scores.sort(key=lambda x: x[1], reverse=True)
    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    emb_rank = build_rank_map(emb_scores)
    bm25_rank = build_rank_map(bm25_scores)

    # RRF + multiplicative boosts
    n = len(rules)
    scored: list[tuple[str, float]] = []

    for rule in rules:
        base = rrf_from_ranks(
            rule.id,
            emb_rank,
            bm25_rank,
            k=RRF_K,
            n_fallback=n,
        )

        boost = 1.0
        if actor_id and rule.actor_id == actor_id:
            boost += 0.3
        if subjects:
            overlap = set(subjects) & set(rule.subjects or [])
            boost += len(overlap) * 0.2

        scored.append((rule.id, base * boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── PATH 2b: Tool rule scoring ──────────────────────────────


def score_tool_rules(
    rules: list[CompiledRule],
    task_context: str,
    data_context: str,
    skill_text: str,
    task_emb: list[float] | None,
    data_emb: list[float] | None,
    skill_emb: list[float] | None,
) -> list[tuple[str, float]]:
    """Score tool rules: max(skill, context) for both signals.

    RRF(embedding, BM25). No actor/subject boosts.
    """
    if not rules:
        return []

    emb_scores: list[tuple[str, float]] = []
    bm25_scores: list[tuple[str, float]] = []

    for rule in rules:
        rule_emb = getattr(rule, "_embedding", None)

        # Embedding: max(skill, weighted_context)
        skill_sim = 0.0
        ctx_sim = 0.0
        if rule_emb:
            if skill_emb:
                skill_sim = cosine_similarity(
                    skill_emb,
                    rule_emb,
                )
            if task_emb or data_emb:
                ctx_sim = weighted_emb_score(
                    task_emb,
                    data_emb,
                    rule_emb,
                )
        emb_scores.append((rule.id, max(skill_sim, ctx_sim)))

        # BM25: max(skill, weighted_context)
        skill_bm = bm25_score(skill_text, rule.text) if skill_text else 0.0
        ctx_bm = weighted_bm25_score(
            task_context,
            data_context,
            rule.text,
        )
        bm25_scores.append((rule.id, max(skill_bm, ctx_bm)))

    # Rank independently
    emb_scores.sort(key=lambda x: x[1], reverse=True)
    bm25_scores.sort(key=lambda x: x[1], reverse=True)
    emb_rank = build_rank_map(emb_scores)
    bm25_rank = build_rank_map(bm25_scores)

    # RRF (no boosts)
    n = len(rules)
    scored: list[tuple[str, float]] = []
    for rule in rules:
        rrf = rrf_from_ranks(
            rule.id,
            emb_rank,
            bm25_rank,
            k=RRF_K,
            n_fallback=n,
        )
        scored.append((rule.id, rrf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Cross-path RRF fusion ───────────────────────────────────


def cross_path_rrf(
    path1: list[tuple[str, float]],
    path2: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge two ranked lists via RRF. Returns [(id, score)]."""
    p1_rank = build_rank_map(path1)
    p2_rank = build_rank_map(path2)

    all_ids = set(p1_rank.keys()) | set(p2_rank.keys())
    n = max(len(p1_rank), len(p2_rank), 1)

    scored = [(rid, rrf_from_ranks(rid, p1_rank, p2_rank, n_fallback=n)) for rid in all_ids]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
