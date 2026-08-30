"""Walk LED_TO chain via recursive queries."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forge_dc.shared.models import ChainNode
from forge_dc.store.models import (
    DecisionNode,
    Document,
    Edge,
)


async def walk_chain(
    session: AsyncSession,
    node_id: str,
    direction: str = "forward",
    max_depth: int = 10,
) -> list[ChainNode]:
    """Walk LED_TO chain in the given direction.

    Uses iterative queries (recursive CTEs not
    available in all SQLite versions).
    """
    visited: set[str] = set()
    chain: list[ChainNode] = []

    # Start node
    start = await _get_node_info(session, node_id)
    if not start:
        return []

    chain.append(start)
    visited.add(node_id)

    # Walk
    current_id = node_id
    for depth in range(1, max_depth + 1):
        next_id = await _get_next_node(session, current_id, direction)
        if not next_id or next_id in visited:
            break

        node_info = await _get_node_info(session, next_id)
        if not node_info:
            break

        node_info.depth = depth
        chain.append(node_info)
        visited.add(next_id)
        current_id = next_id

    return chain


async def _get_node_info(
    session: AsyncSession,
    node_id: str,
) -> ChainNode | None:
    """Get node info for chain display."""
    stmt = (
        select(
            DecisionNode.id,
            DecisionNode.action_label,
            DecisionNode.outcome_short,
            DecisionNode.subjects,
            Document.actor_id,
            Document.actor_type,
        )
        .join(
            Document,
            DecisionNode.document_id == Document.id,
        )
        .where(DecisionNode.id == node_id)
    )
    result = await session.execute(stmt)
    row = result.first()

    if not row:
        return None

    subjects = row[3] or []
    if isinstance(subjects, list):
        subjects = [s if isinstance(s, str) else str(s) for s in subjects]

    return ChainNode(
        id=row[0],
        action_label=row[1],
        outcome_short=row[2] or "",
        subjects=subjects,
        actor_id=row[4],
        actor_type=row[5],
        depth=0,
    )


async def _get_next_node(
    session: AsyncSession,
    current_id: str,
    direction: str,
) -> str | None:
    """Get the next node in the LED_TO chain."""
    if direction == "forward":
        stmt = (
            select(Edge.to_id)
            .where(
                Edge.edge_type == "LED_TO",
                Edge.from_id == current_id,
            )
            .limit(1)
        )
    else:
        stmt = (
            select(Edge.from_id)
            .where(
                Edge.edge_type == "LED_TO",
                Edge.to_id == current_id,
            )
            .limit(1)
        )

    result = await session.execute(stmt)
    row = result.first()
    return row[0] if row else None
