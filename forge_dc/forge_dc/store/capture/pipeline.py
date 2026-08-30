"""Hybrid capture pipeline orchestrator.

H7: Deterministic classify_against_summary before LLM.
H8: Summary updated on correction via update_summary.
H13: Delete includes evidence IDs in edge cleanup.
H14: Dead code removed from delete_document.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from openai import AsyncOpenAI
from sqlalchemy import select

from forge_dc.config import OpenAIConfig
from forge_dc.interface import (
    CaptureStrategy,
)
from forge_dc.shared.aftermath import (
    classify_followup,
    classify_followup_deterministic,
)
from forge_dc.shared.embedder import Embedder
from forge_dc.store.capture.annotator import (
    annotate_nodes,
)
from forge_dc.store.capture.edge_wirer import (
    get_summary_for_doc,
    update_summary,
    wire_edges,
)
from forge_dc.store.capture.evidence import (
    store_evidence,
)
from forge_dc.store.capture.labeler import (
    assign_reasoning_labels,
)
from forge_dc.store.capture.singletons import (
    upsert_singletons,
)
from forge_dc.store.capture.splitter import (
    split_message_history,
)
from forge_dc.store.engine import (
    StoreEngine,
)
from forge_dc.store.models import (
    DecisionNode,
    Document,
    Edge,
    Evidence,
)

logger = logging.getLogger(__name__)


class StoreCapture(CaptureStrategy):
    """Capture strategy for the hybrid approach."""

    def __init__(
        self,
        db_url: str,
        openai_config: OpenAIConfig,
    ) -> None:
        self._engine = StoreEngine(db_url)
        self._openai_config = openai_config
        self._embedder: Embedder | None = None
        self._openai_client: AsyncOpenAI | None = None

    async def initialize(self) -> None:
        await self._engine.create_tables()
        if self._openai_config.api_key:
            self._embedder = Embedder(
                api_key=self._openai_config.api_key,
                model=self._openai_config.embedding_model,
            )
            self._openai_client = AsyncOpenAI(
                api_key=self._openai_config.api_key,
            )

    async def shutdown(self) -> None:
        await self._engine.dispose()

    async def _post_store_hook(
        self,
        session: object,
        doc: object,
        annotated: list[dict],
        message_history: list[dict],
        outcome_summary: str,
        skill_name: str,
        skill_version: str,
        actor_id: str,
        subjects: list[str],
        conversation_id: str | None,
        parent_doc_id: str | None,
        doc_id: str,
        timestamp: datetime,
    ) -> dict | None:
        """Extension point for post-store processing.

        Called after Step 10 (upsert_singletons) and before
        Step 11 (lazy aftermath). Override in subclasses.
        Returns optional dict merged into capture result.
        """
        return None

    async def capture(self, trace: dict) -> dict:
        """Run the full hybrid capture pipeline."""
        # Step 1-2: Extract metadata
        doc_id = str(uuid.uuid4())
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

        subject_ids = [s["id"] if isinstance(s, dict) else s for s in subjects_raw]

        # Step 3: outcome_summary already provided
        # or generate from last assistant message
        if not outcome_summary:
            outcome_summary = _generate_outcome_summary(
                skill_name,
                subject_ids,
                message_history,
            )

        # Step 4: Split into reasoning + tool nodes
        raw_nodes = split_message_history(message_history)
        assign_reasoning_labels(raw_nodes)

        # Step 5: Annotate each node
        annotated = annotate_nodes(raw_nodes, subject_ids)

        # Step 7: Compute embeddings (node + accumulated)
        if self._embedder:
            # Node embeddings
            texts = [a["content"] or " " for a in annotated]
            embeddings = await self._embedder.embed_batch(texts)

            # Accumulated embeddings: content up to each node
            accum_texts = []
            running = ""
            for a in annotated:
                content = a["content"] or " "
                running = f"{running}\n{content}" if running else content
                accum_texts.append(running[:2000])
            accum_embeddings = await self._embedder.embed_batch(
                accum_texts,
            )
        else:
            embeddings = [None] * len(annotated)
            accum_embeddings = [None] * len(annotated)

        # Steps 6, 8-12: Store everything in one txn
        async with self._engine.session() as session:
            # Step 8: Store document
            doc = Document(
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

            # Store DecisionNodes
            node_ids: list[str] = []
            node_types: list[str] = []
            evidence_ids: dict[str, str] = {}
            last_outcome = ""

            for i, ann in enumerate(annotated):
                node_id = str(uuid.uuid4())
                node_ids.append(node_id)
                node_types.append(ann["node_type"])

                # H8: Synthetic per-node valid_time offset
                node = DecisionNode(
                    id=node_id,
                    document_id=doc_id,
                    node_type=ann["node_type"],
                    content=ann["content"],
                    embedding=embeddings[i],
                    accumulated_embedding=accum_embeddings[i],
                    action_label=ann["action_label"],
                    tool_name=ann["tool_name"],
                    subjects=ann["subjects"],
                    evidence_summary=ann["evidence_summary"],
                    outcome_short=ann["outcome_short"],
                    decision_index=ann["decision_index"],
                    valid_time=timestamp + timedelta(seconds=i),
                )
                session.add(node)
                last_outcome = ann["outcome_short"] or ""

                # Step 6: Store evidence for tool nodes
                ev_content = ann.get("evidence_content")
                if ann["node_type"] == "tool" and ev_content:
                    ev_id = await store_evidence(
                        session,
                        node_id,
                        ev_content,
                        ann["tool_name"],
                        timestamp,
                    )
                    evidence_ids[node_id] = ev_id

            # Step 9: Wire edges
            edges_created = await wire_edges(
                session=session,
                document_id=doc_id,
                actor_id=actor_id,
                subjects=subject_ids,
                skill_name=skill_name,
                skill_version=skill_version,
                node_ids=node_ids,
                node_types=node_types,
                evidence_ids=evidence_ids,
                parent_doc_id=parent_doc_id,
                outcome_short=last_outcome,
            )

            # Step 10: Upsert singletons
            singleton_counts = await upsert_singletons(
                session=session,
                actor_id=actor_id,
                actor_name=actor_id,
                actor_type=actor_type,
                subjects=(
                    subjects_raw
                    if subjects_raw and isinstance(subjects_raw[0], dict)
                    else [{"id": s} for s in subjects_raw]
                ),
                skill_name=skill_name,
                skill_version=skill_version,
            )

            # Step 10.5: Post-store hook (extension point)
            hook_result = await self._post_store_hook(
                session=session,
                doc=doc,
                annotated=annotated,
                message_history=message_history,
                outcome_summary=outcome_summary,
                skill_name=skill_name,
                skill_version=skill_version,
                actor_id=actor_id,
                subjects=subject_ids,
                conversation_id=conversation_id,
                parent_doc_id=parent_doc_id,
                doc_id=doc_id,
                timestamp=timestamp,
            )

            # Step 11: Lazy aftermath for previous doc
            aftermath_updated = None
            if conversation_id:
                aftermath_updated = await _compute_lazy_aftermath(
                    session,
                    doc_id,
                    conversation_id,
                    outcome_summary,
                    message_history,
                    self._openai_client,
                    new_skill=skill_name,
                    explicit_parent_id=parent_doc_id,
                )

            # Step 12: Wire FOLLOWED_BY
            if conversation_id and aftermath_updated:
                await _wire_followed_by(
                    session,
                    aftermath_updated["id"],
                    node_ids[0] if node_ids else None,
                )

        result = {
            "document_id": doc_id,
            "nodes_created": len(node_ids),
            "evidence_created": len(evidence_ids),
            "edges_created": edges_created,
            "singletons_upserted": singleton_counts,
            "aftermath_updated": aftermath_updated,
            "timestamp": timestamp.isoformat(),
        }
        if hook_result:
            result.update(hook_result)
        return result

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and all related data.

        H13: Also deletes edges pointing TO evidence
             from other documents.
        H14: Removed dead code (unused query results).

        Returns the number of decision nodes deleted.
        """
        async with self._engine.session() as session:
            # Get node IDs for this document
            result = await session.execute(
                select(DecisionNode.id).where(DecisionNode.document_id == document_id)
            )
            node_ids = [row[0] for row in result.fetchall()]

            if not node_ids:
                raise LookupError(f"Document {document_id} not found")

            node_id_set = set(node_ids)
            nodes_deleted = len(node_ids)

            # H13: Collect evidence IDs so we can
            # delete edges pointing TO evidence too.
            ev_result = await session.execute(
                select(Evidence).where(Evidence.decision_node_id.in_(node_id_set))
            )
            evidence_rows = ev_result.scalars().all()
            ev_id_set = {e.id for e in evidence_rows}

            # All IDs that could appear in edges
            all_ids = node_id_set | ev_id_set

            # Delete edges referencing nodes or evidence
            edge_result = await session.execute(
                select(Edge).where(Edge.from_id.in_(all_ids) | Edge.to_id.in_(all_ids))
            )
            for edge in edge_result.scalars():
                await session.delete(edge)

            # Delete evidence
            for ev in evidence_rows:
                await session.delete(ev)

            # Delete nodes
            dn_result = await session.execute(
                select(DecisionNode).where(DecisionNode.document_id == document_id)
            )
            for dn in dn_result.scalars():
                await session.delete(dn)

            # Delete document
            doc = await session.get(Document, document_id)
            if doc:
                await session.delete(doc)

        return nodes_deleted


def _generate_outcome_summary(
    skill_name: str,
    subject_ids: list[str],
    messages: list[dict],
) -> str:
    """Deterministic outcome summary from message data."""
    last_assistant = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break

    subject_str = ", ".join(subject_ids) if subject_ids else ""
    summary = f"{skill_name} for {subject_str}: {last_assistant[:200]}"
    return summary


async def _compute_lazy_aftermath(
    session,
    new_doc_id: str,
    conversation_id: str,
    new_outcome: str,
    new_messages: list[dict],
    openai_client: AsyncOpenAI | None,
    new_skill: str = "",
    explicit_parent_id: str | None = None,
) -> str | None:
    """Compute aftermath for the previous document.

    When explicit_parent_id is provided, use it to find
    the exact parent doc (supports parallel corrections
    of the same original). Otherwise fall back to
    conversation ordering.

    H7: Tries deterministic classify_against_summary
        before falling back to LLM.
    H8: Updates Summary on correction.
    """
    if explicit_parent_id:
        result = await session.execute(
            select(Document).where(
                Document.id == explicit_parent_id,
            )
        )
    else:
        result = await session.execute(
            select(Document)
            .where(
                Document.conversation_id == conversation_id,
                Document.id != new_doc_id,
                Document.aftermath_status == "pending",
            )
            .order_by(Document.timestamp.desc())
            .limit(1)
        )
    prev_doc = result.scalar_one_or_none()
    if not prev_doc:
        return None

    # Get human message from new doc
    human_message = ""
    for msg in new_messages:
        if msg.get("role") == "user":
            human_message = msg.get("content", "")
            break

    if not human_message:
        return None

    # H7: Try deterministic classification first
    # via shared heuristic, then Summary content,
    # before falling back to LLM.
    prev_summary = await get_summary_for_doc(session, prev_doc.id)
    classification: str | None = None
    needs_context: bool | None = None

    classification = classify_followup_deterministic(
        human_message,
    )
    if classification is None and (prev_summary and prev_summary.content):
        classification = _classify_against_summary(
            human_message=human_message,
            summary_content=prev_summary.content,
        )

    # Fall back to LLM if no deterministic result.
    # The LLM also classifies ambiguity in the same call.
    if classification is None:
        if openai_client:
            result = await classify_followup(
                openai_client,
                human_message=human_message,
                prev_outcome=(prev_doc.outcome_summary or ""),
                prev_skill=prev_doc.skill_name or "",
                new_skill=new_skill,
            )
            classification = result.classification
            needs_context = result.needs_context
        else:
            classification = "new_request"

    if classification == "correction":
        prev_doc.aftermath_status = "corrected"
        prev_doc.correction_text = human_message

        # Set parent_doc_id on the new doc so LED_TO cross_run
        # edges can be created. The aftermath discovered the
        # parent — make it explicit.
        new_doc_result = await session.execute(select(Document).where(Document.id == new_doc_id))
        new_doc = new_doc_result.scalar_one_or_none()
        if new_doc and not new_doc.parent_doc_id:
            new_doc.parent_doc_id = prev_doc.id

        # Determine if correction needs context.
        # LLM result takes precedence; fall back to
        # deterministic heuristic.
        if needs_context is not None:
            prev_doc.correction_needs_context = needs_context
        else:
            from forge_dc.shared.ambiguity import (
                correction_needs_context,
            )

            prev_doc.correction_needs_context = correction_needs_context(human_message)

        # H8: Update Summary on correction
        if prev_summary:
            await update_summary(
                session,
                prev_summary.id,
                human_message,
            )
    else:
        prev_doc.aftermath_status = "accepted"

    return {"id": prev_doc.id, "status": prev_doc.aftermath_status}


def _classify_against_summary(
    human_message: str,
    summary_content: str,
) -> str | None:
    """Deterministic classification using Summary.

    H7: Simple heuristic — if the human message
    references keywords that suggest correction
    (e.g. "wrong", "fix", "change", "actually",
    "instead", "correct"), classify as correction.
    Otherwise classify as new_request.

    Returns None if we can't determine confidently
    (falls back to LLM).
    """
    msg_lower = human_message.lower()
    correction_signals = [
        "wrong",
        "fix",
        "change",
        "actually",
        "instead",
        "correct",
        "update",
        "should have",
        "should be",
        "not right",
        "redo",
        "revise",
        "include",
        "add",
        "also",
        "missing",
        "forgot",
    ]
    for signal in correction_signals:
        if signal in msg_lower:
            return "correction"

    # If the human message clearly references a
    # different skill/topic, it's a new request.
    # Otherwise fall back to LLM — don't guess.
    return None


async def _wire_followed_by(
    session,
    prev_doc_id: str,
    first_node_id: str | None,
) -> None:
    """Wire FOLLOWED_BY edge if previous chain ended."""
    if not first_node_id:
        return

    prev_doc = await session.get(Document, prev_doc_id)
    if not prev_doc:
        return
    if prev_doc.aftermath_status != "accepted":
        return

    # Get last node of previous doc
    result = await session.execute(
        select(DecisionNode.id)
        .where(DecisionNode.document_id == prev_doc_id)
        .order_by(DecisionNode.decision_index.desc())
        .limit(1)
    )
    prev_last = result.scalar_one_or_none()
    if not prev_last:
        return

    edge = Edge(
        id=str(uuid.uuid4()),
        edge_type="FOLLOWED_BY",
        from_id=prev_last,
        to_id=first_node_id,
    )
    session.add(edge)
