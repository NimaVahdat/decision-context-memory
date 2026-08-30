"""ORM models for this approach — 3 tables.

Tables: documents, chunks, conversations.
No edges, no singletons, no evidence table.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)
from sqlalchemy.types import JSON


class DocBase(DeclarativeBase):
    """Separate base for doc approach tables."""

    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class DocDocument(DocBase):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    skill_name: Mapped[str | None] = mapped_column(String)
    skill_version: Mapped[str | None] = mapped_column(String)
    actor_id: Mapped[str | None] = mapped_column(String)
    actor_type: Mapped[str | None] = mapped_column(String)
    subjects: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        default=list,
    )
    conversation_id: Mapped[str | None] = mapped_column(String)
    parent_doc_id: Mapped[str | None] = mapped_column(String)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=_utcnow)
    outcome_summary: Mapped[str | None] = mapped_column(Text)
    aftermath_status: Mapped[str] = mapped_column(String, default="pending")
    correction_text: Mapped[str | None] = mapped_column(Text)
    correction_needs_context: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    message_history: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
    )

    __table_args__ = (
        Index("idx_doc_docs_conversation", "conversation_id"),
        Index("idx_doc_docs_actor", "actor_id"),
        Index("idx_doc_docs_skill", "skill_name"),
    )


class Chunk(DocBase):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    document_id: Mapped[str] = mapped_column(String, ForeignKey("documents.id"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
    )
    action_label: Mapped[str | None] = mapped_column(String)
    tool_name: Mapped[str | None] = mapped_column(String)
    evidence_summary: Mapped[str | None] = mapped_column(Text)
    outcome_short: Mapped[str | None] = mapped_column(Text)
    decision_index: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_action", "action_label"),
    )


class Conversation(DocBase):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    documents: Mapped[list | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )
