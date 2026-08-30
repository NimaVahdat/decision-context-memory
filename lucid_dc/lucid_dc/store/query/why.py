"""Lazy reason extraction for a DecisionNode."""

from __future__ import annotations

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lucid_dc.shared.models import WhyResult
from lucid_dc.store.models import (
    DecisionNode,
    Edge,
)


async def extract_reason(
    session: AsyncSession,
    node_id: str,
    openai_client: AsyncOpenAI | None = None,
) -> WhyResult:
    """Extract or return cached reason for a node."""
    node = await session.get(DecisionNode, node_id)
    if not node:
        raise LookupError(f"DecisionNode {node_id} not found")

    # Already cached?
    if node.reason:
        return WhyResult(
            node_id=node_id,
            reason=node.reason,
            cached=True,
        )

    # For reasoning nodes, content IS the reason
    if node.node_type == "reasoning":
        reason = node.content
    elif openai_client:
        # Gather adjacent reasoning nodes for context
        adjacent_content = await _get_adjacent_reasoning(session, node_id)
        context = "\n---\n".join(adjacent_content + [node.content])
        reason = await _llm_extract_reason(
            openai_client,
            context=context,
            action=node.action_label or "",
        )
    else:
        # No LLM available — use content as-is
        reason = node.content

    # Cache for future requests
    node.reason = reason

    return WhyResult(
        node_id=node_id,
        reason=reason,
        cached=False,
    )


async def _get_adjacent_reasoning(
    session: AsyncSession,
    node_id: str,
) -> list[str]:
    """Get content from adjacent reasoning nodes."""
    # Find nodes connected via LED_TO (either direction)
    adjacent_ids: list[str] = []

    # Incoming LED_TO
    stmt = select(Edge.from_id).where(
        Edge.edge_type == "LED_TO",
        Edge.to_id == node_id,
    )
    result = await session.execute(stmt)
    for row in result.fetchall():
        adjacent_ids.append(row[0])

    # Outgoing LED_TO
    stmt = select(Edge.to_id).where(
        Edge.edge_type == "LED_TO",
        Edge.from_id == node_id,
    )
    result = await session.execute(stmt)
    for row in result.fetchall():
        adjacent_ids.append(row[0])

    # Fetch reasoning nodes only
    contents = []
    for adj_id in adjacent_ids:
        adj_node = await session.get(DecisionNode, adj_id)
        if adj_node and adj_node.node_type == "reasoning":
            contents.append(adj_node.content)

    return contents


async def _llm_extract_reason(
    client: AsyncOpenAI,
    context: str,
    action: str,
) -> str:
    """Extract reason from context using LLM."""
    prompt = (
        "Given the following agent decision context, "
        "extract a concise reason (1-2 sentences) "
        f"explaining why the agent took the action "
        f"'{action}'.\n\n"
        f"Context:\n{context}\n\n"
        "Reason:"
    )

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=150,
    )

    from lucid_dc.shared.usage import record_llm

    usage = getattr(response, "usage", None)
    if usage is not None:
        record_llm(
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )

    return (response.choices[0].message.content or "").strip()
