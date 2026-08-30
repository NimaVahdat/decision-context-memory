"""Naive doc capture — chunk by tool groups, embed, store.

No annotations, no aftermath classification, no conversation
tracking. The simplest possible capture pipeline.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from doc_chunk.capture import (
    _chunk_by_tool_groups,
)
from doc_chunk.config import OpenAIConfig
from doc_chunk.engine import DocEngine
from doc_chunk.interface import CaptureStrategy
from doc_chunk.models import (
    Chunk,
    DocDocument,
)
from doc_chunk.shared.embedder import Embedder

logger = logging.getLogger(__name__)


class DocNaiveCapture(CaptureStrategy):
    """Dump-and-pray capture: chunk, embed, store."""

    def __init__(
        self,
        db_url: str,
        openai_config: OpenAIConfig,
    ) -> None:
        self._engine = DocEngine(db_url)
        self._openai_config = openai_config
        self._embedder: Embedder | None = None

    async def initialize(self) -> None:
        await self._engine.create_tables()
        if self._openai_config.api_key:
            self._embedder = Embedder(
                api_key=self._openai_config.api_key,
                model=self._openai_config.embedding_model,
            )

    async def shutdown(self) -> None:
        await self._engine.dispose()

    async def capture(self, trace: dict) -> dict:
        """Chunk by tool groups, embed, store. Nothing else."""
        doc_id = str(uuid.uuid4())

        # Parse metadata
        skill_name = trace.get("skill_name", "")
        skill_version = trace.get("skill_version", "")
        actor_id = trace.get("actor_id", "")
        actor_type = trace.get("actor_type", "agent")
        subjects_raw = trace.get("subjects", [])
        conversation_id = trace.get("conversation_id")
        parent_doc_id = trace.get("parent_doc_id")
        timestamp_str = trace.get("timestamp")
        outcome_summary = trace.get("outcome_summary", "")
        message_history = trace.get("message_history", [])

        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now(UTC)

        # Chunk at tool-call boundaries
        raw_chunks = _chunk_by_tool_groups(message_history)

        # Build plain content strings from each chunk
        contents = [_assemble_content(chunk) for chunk in raw_chunks]

        # Embed
        if self._embedder:
            embeddings = await self._embedder.embed_batch(
                contents,
            )
        else:
            embeddings = [None] * len(contents)

        # Store document + chunks
        async with self._engine.session() as session:
            doc = DocDocument(
                id=doc_id,
                skill_name=skill_name,
                skill_version=skill_version,
                actor_id=actor_id,
                actor_type=actor_type,
                subjects=[s if isinstance(s, dict) else {"id": s} for s in subjects_raw],
                conversation_id=conversation_id,
                parent_doc_id=parent_doc_id,
                timestamp=timestamp,
                outcome_summary=outcome_summary,
                aftermath_status="pending",
                message_history=message_history,
            )
            session.add(doc)
            await session.flush()

            for i, content in enumerate(contents):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    content=content,
                    chunk_index=i,
                    embedding=embeddings[i],
                    action_label=None,
                    tool_name=None,
                    evidence_summary=None,
                    outcome_short=None,
                    decision_index=None,
                )
                session.add(chunk)

        return {
            "document_id": doc_id,
            "chunks_created": len(contents),
        }

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and its chunks.

        Returns the number of chunks deleted.
        """
        from sqlalchemy import select

        async with self._engine.session() as session:
            result = await session.execute(
                select(Chunk).where(
                    Chunk.document_id == document_id,
                )
            )
            chunks = result.scalars().all()
            chunks_deleted = len(chunks)
            for chunk in chunks:
                await session.delete(chunk)

            doc = await session.get(DocDocument, document_id)
            if not doc:
                raise LookupError(f"Document {document_id} not found")
            await session.delete(doc)

        return chunks_deleted


def _assemble_content(chunk: dict) -> str:
    """Build a plain text string from chunk messages.

    Dump-and-pray: include ALL message content with no
    truncation. User messages, tool calls, full tool
    results, assistant responses — everything.
    """
    if "split_content" in chunk:
        return chunk["split_content"]

    parts: list[str] = []
    for msg in chunk["messages"]:
        role = msg.get("role", "")
        text = msg.get("content", "")
        if role == "user":
            parts.append(f"User: {text}")
        elif role == "assistant":
            if text.strip():
                parts.append(f"Assistant: {text}")
        elif role == "tool_call":
            name = msg.get("name", "")
            args = msg.get("arguments", {})
            parts.append(f"Tool call: {name}({args})")
        elif role == "tool_result":
            parts.append(f"Tool result:\n{text}")
    return "\n\n".join(parts)
