"""Deterministic edge wiring — all 9 edge types.

H6: update_summary helper for correction updates.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lucid_dc.store.models import (
    DecisionNode,
    Edge,
    Summary,
)


async def wire_edges(
    session: AsyncSession,
    document_id: str,
    actor_id: str,
    subjects: list[str],
    skill_name: str,
    skill_version: str,
    node_ids: list[str],
    node_types: list[str],
    evidence_ids: dict[str, str],
    parent_doc_id: str | None,
    outcome_short: str,
) -> int:
    """Wire all edges for a captured document.

    Returns the number of edges created.
    """
    edges_created = 0
    policy_id = f"{skill_name}-{skill_version}"

    # 9a. MADE — Actor → each DecisionNode
    for node_id in node_ids:
        _add_edge(session, "MADE", actor_id, node_id)
        edges_created += 1

    # 9b. ABOUT — DecisionNode → Subject
    for node_id in node_ids:
        for subject_id in subjects:
            _add_edge(session, "ABOUT", node_id, subject_id)
            edges_created += 1

    # 9c. APPLIED — DecisionNode → Policy
    for node_id in node_ids:
        _add_edge(session, "APPLIED", node_id, policy_id)
        edges_created += 1

    # 9d. LED_TO — Sequential within run
    for i in range(1, len(node_ids)):
        _add_edge(
            session,
            "LED_TO",
            node_ids[i - 1],
            node_ids[i],
            metadata={"type": "intra_run"},
        )
        edges_created += 1

    # 9e. LED_TO — Cross-run (delegation)
    if parent_doc_id:
        parent_last = await _get_last_node_id(session, parent_doc_id)
        if parent_last and node_ids:
            _add_edge(
                session,
                "LED_TO",
                parent_last,
                node_ids[0],
                metadata={"type": "cross_run"},
            )
            edges_created += 1

    # 9f. PRODUCED — Tool node → Evidence
    for node_id, ev_id in evidence_ids.items():
        _add_edge(session, "PRODUCED", node_id, ev_id)
        edges_created += 1

    # 9g. INFORMED_BY — Current node → prev Evidence
    for i in range(1, len(node_ids)):
        prev_id = node_ids[i - 1]
        if prev_id in evidence_ids:
            _add_edge(
                session,
                "INFORMED_BY",
                node_ids[i],
                evidence_ids[prev_id],
            )
            edges_created += 1

    # 9h. HAS_SUMMARY — Last node → Summary
    if node_ids:
        subject_str = ", ".join(subjects) if subjects else ""
        summary_content = f"{skill_name} for {subject_str}: {outcome_short}"
        summary_id = str(uuid.uuid4())
        summary = Summary(
            id=summary_id,
            content=summary_content,
        )
        session.add(summary)
        _add_edge(
            session,
            "HAS_SUMMARY",
            node_ids[-1],
            summary_id,
        )
        edges_created += 1

    return edges_created


async def update_summary(
    session: AsyncSession,
    summary_id: str,
    correction_text: str,
) -> None:
    """Update a Summary's content with correction.

    H6/H8: When aftermath is classified as corrected,
    append the correction text to the Summary content.
    """
    summary = await session.get(Summary, summary_id)
    if not summary:
        return
    summary.content = f"{summary.content or ''}\n[corrected]: {correction_text}"
    summary.last_updated = datetime.now(UTC)


async def get_summary_for_doc(
    session: AsyncSession,
    document_id: str,
) -> Summary | None:
    """Get Summary for a document via HAS_SUMMARY edge.

    Finds the last node of the document, then follows
    the HAS_SUMMARY edge to the Summary row.
    """
    last_node_id = await _get_last_node_id(session, document_id)
    if not last_node_id:
        return None

    result = await session.execute(
        select(Edge.to_id).where(
            Edge.edge_type == "HAS_SUMMARY",
            Edge.from_id == last_node_id,
        )
    )
    summary_id = result.scalar_one_or_none()
    if not summary_id:
        return None

    return await session.get(Summary, summary_id)


def _add_edge(
    session: AsyncSession,
    edge_type: str,
    from_id: str,
    to_id: str,
    metadata: dict | None = None,
) -> None:
    edge = Edge(
        id=str(uuid.uuid4()),
        edge_type=edge_type,
        from_id=from_id,
        to_id=to_id,
        metadata_=metadata,
    )
    session.add(edge)


async def _get_last_node_id(
    session: AsyncSession,
    document_id: str,
) -> str | None:
    """Get the last DecisionNode ID for a document."""
    result = await session.execute(
        select(DecisionNode.id)
        .where(DecisionNode.document_id == document_id)
        .order_by(DecisionNode.decision_index.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return row
