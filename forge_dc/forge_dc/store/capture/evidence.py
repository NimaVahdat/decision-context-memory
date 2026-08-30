"""Store full tool output in the Evidence table."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from forge_dc.store.models import (
    Evidence,
)


async def store_evidence(
    session: AsyncSession,
    node_id: str,
    evidence_content: str,
    tool_name: str | None,
    valid_time: datetime | None,
) -> str:
    """Store full evidence for a tool node. Returns evidence ID."""
    evidence_id = str(uuid.uuid4())
    evidence = Evidence(
        id=evidence_id,
        decision_node_id=node_id,
        content=evidence_content,
        source=tool_name,
        produced_at=valid_time or datetime.now(UTC),
    )
    session.add(evidence)
    return evidence_id
