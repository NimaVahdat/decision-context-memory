"""Layer 3 — Skill-level aggregation (global readout)."""

from __future__ import annotations

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from forge_dc.store.models import (
    DecisionNode,
    Document,
)


async def compute_global_readout(
    session: AsyncSession,
    skill_name: str,
    min_runs: int = 3,
    actor_id: str | None = None,
) -> dict:
    """Compute skill-level stats for Layer 3 injection.

    Returns aggregated action stats: runs, corrections,
    correction_patterns, outcome_variety, is_settled.

    When actor_id is set, correction patterns are
    annotated with ownership so the formatting layer
    can distinguish the current user's patterns.
    """
    # Query: group by action_label, compute stats
    stmt = (
        select(
            DecisionNode.action_label,
            func.count().label("runs"),
            func.count().filter(Document.aftermath_status == "corrected").label("corrections"),
            func.count(distinct(DecisionNode.outcome_short)).label("outcome_variety"),
        )
        .join(
            Document,
            DecisionNode.document_id == Document.id,
        )
        .where(Document.skill_name == skill_name)
        .group_by(DecisionNode.action_label)
        .having(func.count() >= min_runs)
        .order_by(func.count().filter(Document.aftermath_status == "corrected").desc())
    )

    result = await session.execute(stmt)
    rows = result.fetchall()

    # Get total runs
    total_stmt = select(func.count(distinct(Document.id))).where(Document.skill_name == skill_name)
    total_result = await session.execute(total_stmt)
    total_runs = total_result.scalar() or 0

    # Get correction patterns per action
    actions = []
    for row in rows:
        action_label = row[0]
        runs = row[1]
        corrections = row[2]
        outcome_variety = row[3]

        # Fetch correction patterns (with actor info
        # when actor_id is set).
        patterns_stmt = (
            select(
                Document.correction_text,
                Document.actor_id,
            )
            .join(
                DecisionNode,
                DecisionNode.document_id == Document.id,
            )
            .where(
                Document.skill_name == skill_name,
                DecisionNode.action_label == action_label,
                Document.correction_text.isnot(None),
            )
        )
        patterns_result = await session.execute(
            patterns_stmt,
        )

        # Deduplicate by correction_text in Python.
        # Keep the first actor_id seen per text.
        seen_texts: dict[str, str | None] = {}
        for p in patterns_result.fetchall():
            text, doc_actor = p[0], p[1]
            if text and text not in seen_texts:
                seen_texts[text] = doc_actor

        # When actor_id is set, annotate patterns
        # with ownership for the formatting layer.
        if actor_id:
            correction_patterns = []
            for text, doc_actor in seen_texts.items():
                if doc_actor == actor_id:
                    correction_patterns.append(
                        f"[yours] {text}",
                    )
                else:
                    correction_patterns.append(
                        f"[other user] {text}",
                    )
        else:
            correction_patterns = list(seen_texts.keys())

        is_settled = runs >= 10 and corrections == 0 and outcome_variety <= 3

        correction_rate = corrections / runs if runs > 0 else 0.0

        actions.append(
            {
                "action_label": action_label,
                "runs": runs,
                "corrections": corrections,
                "correction_rate": round(correction_rate, 3),
                "correction_patterns": (correction_patterns),
                "outcome_variety": outcome_variety,
                "is_settled": is_settled,
            }
        )

    return {
        "skill_name": skill_name,
        "total_runs": total_runs,
        "actions": actions,
    }
