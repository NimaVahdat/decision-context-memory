"""ORM models for the hybrid approach — 8 tables.

Tables: documents, decision_nodes, evidence, actors,
subjects, policies, edges, summaries.
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


class HybridBase(DeclarativeBase):
    """Separate base for hybrid approach tables."""

    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ── Singletons ───────────────────────────────────────


class Actor(HybridBase):
    __tablename__ = "actors"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    type: Mapped[str | None] = mapped_column(String)


class Subject(HybridBase):
    __tablename__ = "subjects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)


class Policy(HybridBase):
    __tablename__ = "policies"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str | None] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)


# ── Documents (run-level metadata) ───────────────────


class Document(HybridBase):
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
        Index("idx_docs_conversation", "conversation_id"),
        Index("idx_docs_actor", "actor_id"),
        Index("idx_docs_skill", "skill_name"),
    )


# ── Decision Nodes (the core unit) ──────────────────


class DecisionNode(HybridBase):
    __tablename__ = "decision_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    document_id: Mapped[str] = mapped_column(String, ForeignKey("documents.id"))
    node_type: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Embedding stored as JSON array for SQLite compat.
    # For production PostgreSQL, use pgvector.
    embedding: Mapped[list | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
    )
    # Accumulated embedding: all content up to and including
    # this node in the trace. For stage-to-stage matching.
    accumulated_embedding: Mapped[list | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
    )
    action_label: Mapped[str | None] = mapped_column(String)
    tool_name: Mapped[str | None] = mapped_column(String)
    # Subjects stored as JSON array (SQLite compat).
    # PostgreSQL would use ARRAY(String).
    subjects: Mapped[list | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
    )
    evidence_summary: Mapped[str | None] = mapped_column(Text)
    outcome_short: Mapped[str | None] = mapped_column(Text)
    decision_index: Mapped[int | None] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text)
    valid_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transaction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("idx_dn_document", "document_id"),
        Index("idx_dn_action", "action_label"),
        Index("idx_dn_type", "node_type"),
    )


# ── Evidence (full tool output) ──────────────────────


class Evidence(HybridBase):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    decision_node_id: Mapped[str] = mapped_column(String, ForeignKey("decision_nodes.id"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String)
    produced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("idx_evidence_node", "decision_node_id"),)


# ── Edges (generic edge table) ──────────────────────


class Edge(HybridBase):
    __tablename__ = "edges"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    edge_type: Mapped[str] = mapped_column(String, nullable=False)
    from_id: Mapped[str] = mapped_column(String, nullable=False)
    to_id: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
    )

    __table_args__ = (
        Index("idx_edges_type_from", "edge_type", "from_id"),
        Index("idx_edges_type_to", "edge_type", "to_id"),
    )


# ── Summaries (chain classification) ────────────────


class Summary(HybridBase):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_uuid)
    content: Mapped[str | None] = mapped_column(Text)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
