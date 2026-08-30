"""Hybrid query strategy — search, global,
precedents, chain, why, context.

H4:  POST /query/context endpoint (1-hop subgraph).
H15: Structured per-node edge format in get_document.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI
from sqlalchemy import select

from lucid_dc.config import OpenAIConfig
from lucid_dc.interface import (
    QueryStrategy,
)
from lucid_dc.shared.embedder import Embedder
from lucid_dc.shared.models import (
    ChainNode,
    WhyResult,
)
from lucid_dc.shared.search import (
    bm25_score,
    cosine_similarity,
    reciprocal_rank_fusion,
)
from lucid_dc.store.engine import (
    StoreEngine,
)
from lucid_dc.store.models import (
    Actor,
    DecisionNode,
    Document,
    Edge,
    Evidence,
    Policy,
    Subject,
)
from lucid_dc.store.query.chain import (
    walk_chain,
)
from lucid_dc.store.query.global_readout import (
    compute_global_readout,
)
from lucid_dc.store.query.precedents import (
    query_evidence_corrections,
    query_precedents,
)
from lucid_dc.store.query.why import (
    extract_reason,
)

logger = logging.getLogger(__name__)


class StoreQuery(QueryStrategy):
    """Query strategy for the hybrid approach."""

    @property
    def format_mode(self) -> str:
        return "adaptive"

    @property
    def supports_global_readout(self) -> bool:
        return True

    @property
    def supports_evidence_corrections(self) -> bool:
        return True

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

    async def global_readout(
        self,
        skill_name: str,
        min_runs: int,
        actor_id: str | None = None,
    ) -> dict:
        async with self._engine.session() as session:
            return await compute_global_readout(
                session,
                skill_name,
                min_runs,
                actor_id=actor_id,
            )

    async def search(
        self,
        query_text: str,
        subjects: list[str] | None = None,
        skill_name: str | None = None,
        action_label: str | None = None,
        aftermath_filter: str | None = None,
        search_mode: str = "hybrid",
        result_mode: str = "chunk",
        limit: int = 5,
    ) -> list[dict]:
        """Hybrid search: BM25 + vector + RRF.

        result_mode is accepted for interface compat but
        hybrid always returns node-level results (edges
        provide document context via chain walk).
        """
        query_embedding = None
        if self._embedder:
            query_embedding = await self._embedder.embed(query_text)

        async with self._engine.session() as session:
            return await _hybrid_search(
                session=session,
                query_text=query_text,
                query_embedding=query_embedding,
                subjects=subjects,
                skill_name=skill_name,
                action_label=action_label,
                aftermath_filter=aftermath_filter,
                search_mode=search_mode,
                limit=limit,
            )

    async def precedents(
        self,
        description: str,
        subjects: list[str] | None = None,
        evidence: str | None = None,
        limit: int = 5,
        actor_id: str | None = None,
    ) -> dict:
        async with self._engine.session() as session:
            return await query_precedents(
                session=session,
                description=description,
                subjects=subjects,
                evidence=evidence,
                limit=limit,
                embedder=self._embedder,
                actor_id=actor_id,
            )

    async def evidence_corrections(
        self,
        tool_result: str,
        skill_name: str,
        limit: int = 5,
        subjects: list[str] | None = None,
    ) -> list[dict]:
        """Find corrections via INFORMED_BY evidence.

        Feature 2: queries Evidence table for content
        similar to the tool result, follows INFORMED_BY
        edges back to corrected DecisionNodes. Uses hybrid
        retrieval: BM25 over Evidence.content combined via
        RRF with cosine similarity over DecisionNode
        embeddings, plus a subject overlap boost.
        """
        async with self._engine.session() as session:
            return await query_evidence_corrections(
                session=session,
                tool_result=tool_result,
                skill_name=skill_name,
                limit=limit,
                embedder=self._embedder,
                subjects=subjects,
            )

    async def chain(
        self,
        node_id: str,
        direction: str = "forward",
        max_depth: int = 10,
    ) -> list[ChainNode]:
        async with self._engine.session() as session:
            return await walk_chain(session, node_id, direction, max_depth)

    async def why(self, node_id: str) -> WhyResult:
        async with self._engine.session() as session:
            return await extract_reason(
                session,
                node_id,
                self._openai_client,
            )

    async def get_document(self, document_id: str) -> dict:
        """Retrieve a full document with all nodes.

        H15: Edges are structured per-node objects
        instead of flat lists.
        """
        async with self._engine.session() as session:
            doc = await session.get(Document, document_id)
            if not doc:
                raise LookupError(f"Document {document_id} not found")

            nodes_result = await session.execute(
                select(DecisionNode)
                .where(DecisionNode.document_id == document_id)
                .order_by(DecisionNode.decision_index)
            )
            nodes = nodes_result.scalars().all()

            node_ids = [n.id for n in nodes]
            evidence_result = await session.execute(
                select(Evidence).where(Evidence.decision_node_id.in_(node_ids))
            )
            evidence_map = {
                e.decision_node_id: {
                    "id": e.id,
                    "content": e.content,
                    "source": e.source,
                }
                for e in evidence_result.scalars()
            }

            # H15: Build structured edge maps per node
            edges_result = await session.execute(
                select(Edge).where(Edge.from_id.in_(node_ids) | Edge.to_id.in_(node_ids))
            )
            structured_edges = _build_structured_edges(
                node_ids,
                list(edges_result.scalars()),
            )

            return {
                "id": doc.id,
                "skill_name": doc.skill_name,
                "actor_id": doc.actor_id,
                "subjects": doc.subjects,
                "aftermath_status": (doc.aftermath_status),
                "correction_text": doc.correction_text,
                "outcome_summary": doc.outcome_summary,
                "timestamp": (doc.timestamp.isoformat() if doc.timestamp else None),
                "nodes": [
                    {
                        "id": n.id,
                        "node_type": n.node_type,
                        "content": n.content,
                        "action_label": (n.action_label),
                        "tool_name": n.tool_name,
                        "subjects": n.subjects,
                        "evidence_summary": (n.evidence_summary),
                        "outcome_short": (n.outcome_short),
                        "decision_index": (n.decision_index),
                        "reason": n.reason,
                        "evidence": evidence_map.get(n.id),
                        "edges": structured_edges.get(
                            n.id,
                            _empty_edge_struct(),
                        ),
                    }
                    for n in nodes
                ],
            }

    async def context(self, node_id: str) -> dict:
        """1-hop subgraph around a DecisionNode.

        H4: Returns node info + connected actor,
        subjects, policy, parent/children via LED_TO,
        and informed_by evidence.
        """
        async with self._engine.session() as session:
            # Get the node
            node = await session.get(DecisionNode, node_id)
            if not node:
                raise LookupError(f"Node {node_id} not found")

            # Get all edges touching this node
            edges_result = await session.execute(
                select(Edge).where((Edge.from_id == node_id) | (Edge.to_id == node_id))
            )
            edges = edges_result.scalars().all()

            # Extract connected entity IDs
            actor_id = None
            subject_ids: list[str] = []
            policy_id = None
            parent_node_id = None
            child_node_ids: list[str] = []
            informed_by_ids: list[str] = []
            produced_id = None

            for edge in edges:
                if edge.edge_type == "MADE":
                    actor_id = edge.from_id
                elif edge.edge_type == "ABOUT":
                    if edge.from_id == node_id:
                        subject_ids.append(edge.to_id)
                elif edge.edge_type == "APPLIED":
                    if edge.from_id == node_id:
                        policy_id = edge.to_id
                elif edge.edge_type == "LED_TO":
                    if edge.to_id == node_id:
                        parent_node_id = edge.from_id
                    if edge.from_id == node_id:
                        child_node_ids.append(edge.to_id)
                elif edge.edge_type == "INFORMED_BY":
                    if edge.from_id == node_id:
                        informed_by_ids.append(edge.to_id)
                elif edge.edge_type == "PRODUCED" and edge.from_id == node_id:
                    produced_id = edge.to_id

            # Resolve actor details
            actor_info = None
            if actor_id:
                actor = await session.get(Actor, actor_id)
                if actor:
                    actor_info = {
                        "id": actor.id,
                        "name": actor.name,
                        "type": actor.type,
                    }

            # Resolve subject details
            subjects_info = []
            for sid in subject_ids:
                subj = await session.get(Subject, sid)
                if subj:
                    subjects_info.append(
                        {
                            "id": subj.id,
                            "name": subj.name,
                        }
                    )

            # Resolve policy details
            policy_info = None
            if policy_id:
                policy = await session.get(Policy, policy_id)
                if policy:
                    policy_info = {
                        "id": policy.id,
                        "version": policy.version,
                    }

            return {
                "node_id": node.id,
                "node_type": node.node_type,
                "content": node.content,
                "action_label": node.action_label,
                "tool_name": node.tool_name,
                "subjects": node.subjects,
                "evidence_summary": (node.evidence_summary),
                "outcome_short": node.outcome_short,
                "decision_index": node.decision_index,
                "reason": node.reason,
                "actor": actor_info,
                "subject_details": subjects_info,
                "policy": policy_info,
                "parent_node_id": parent_node_id,
                "child_node_ids": child_node_ids,
                "informed_by_ids": informed_by_ids,
                "produced_id": produced_id,
            }

    async def get_conversation_documents(
        self,
        conversation_id: str,
    ) -> dict:
        """Return all documents in a conversation.

        D4: Includes correction_text per document.
        """
        async with self._engine.session() as session:
            result = await session.execute(
                select(Document)
                .where(Document.conversation_id == conversation_id)
                .order_by(Document.timestamp)
            )
            docs = result.scalars().all()
            return {
                "conversation_id": conversation_id,
                "documents": [
                    {
                        "id": d.id,
                        "skill_name": d.skill_name,
                        "actor_id": d.actor_id,
                        "subjects": d.subjects,
                        "aftermath_status": (d.aftermath_status),
                        "outcome_summary": (d.outcome_summary),
                        "correction_text": (d.correction_text),
                        "timestamp": (d.timestamp.isoformat() if d.timestamp else None),
                    }
                    for d in docs
                ],
            }


def _empty_edge_struct() -> dict:
    """Return an empty structured edge object."""
    return {
        "made_by": None,
        "about": [],
        "applied": None,
        "led_to": None,
        "informed_by": None,
        "produced": None,
    }


def _build_structured_edges(
    node_ids: list[str],
    edges: list,
) -> dict[str, dict]:
    """Build H15 structured per-node edge objects.

    Instead of flat [{type, from, to}, ...], returns:
    {
      "made_by": "actor-id",
      "about": ["subj-1", ...],
      "applied": "policy-id",
      "led_to": "next-node-id",
      "informed_by": "evidence-id" | null,
      "produced": "evidence-id" | null,
    }
    """
    result: dict[str, dict] = {}

    for nid in node_ids:
        result[nid] = _empty_edge_struct()

    for edge in edges:
        etype = edge.edge_type
        for nid in node_ids:
            if etype == "MADE" and edge.to_id == nid:
                result[nid]["made_by"] = edge.from_id
            elif etype == "ABOUT" and edge.from_id == nid:
                result[nid]["about"].append(edge.to_id)
            elif etype == "APPLIED" and edge.from_id == nid:
                result[nid]["applied"] = edge.to_id
            elif etype == "LED_TO" and edge.from_id == nid:
                result[nid]["led_to"] = edge.to_id
            elif etype == "INFORMED_BY" and edge.from_id == nid:
                result[nid]["informed_by"] = edge.to_id
            elif etype == "PRODUCED" and edge.from_id == nid:
                result[nid]["produced"] = edge.to_id

    return result


def _extract_subject_names(
    subjects: list | dict | str | None,
) -> list[str]:
    """Extract subject name strings for BM25 scoring."""
    if not subjects:
        return []
    if isinstance(subjects, str):
        subjects = json.loads(subjects)
    if not isinstance(subjects, list):
        return []
    names: list[str] = []
    for s in subjects:
        if isinstance(s, dict):
            name = s.get("name", "")
            if name:
                names.append(name)
        elif isinstance(s, str):
            names.append(s)
    return names


def _subjects_match(
    doc_subjects: list | dict | None,
    filter_ids: list[str],
) -> bool:
    """Check if document subjects contain any filter ID.

    Subjects are stored as JSON (list of dicts or strings).
    """
    if not doc_subjects:
        return False
    subjects = doc_subjects
    if isinstance(subjects, str):
        subjects = json.loads(subjects)
    if not isinstance(subjects, list):
        return False
    for s in subjects:
        if isinstance(s, dict):
            if s.get("id") in filter_ids:
                return True
        elif s in filter_ids:
            return True
    return False


def _build_item(row) -> dict:
    """Build a search result dict from a DB row."""
    return {
        "node_id": row[0],
        "node_type": row[1],
        "content": row[2],
        "action_label": row[3],
        "tool_name": row[4],
        "subjects": row[5] or [],
        "evidence_summary": row[6] or "",
        "outcome_short": row[7] or "",
        "decision_index": row[8] or 0,
        "document": {
            "id": row[10],
            "skill_name": row[11],
            "subjects": row[12],
            "timestamp": (row[13].isoformat() if row[13] else None),
            "aftermath_status": row[14],
            "correction_text": row[15],
        },
    }


async def _hybrid_search(
    session,
    query_text: str,
    query_embedding: list[float] | None,
    subjects: list[str] | None,
    skill_name: str | None,
    action_label: str | None,
    aftermath_filter: str | None,
    search_mode: str,
    limit: int,
) -> list[dict]:
    """Search decision nodes with BM25 + vector + RRF."""
    stmt = select(
        DecisionNode.id,
        DecisionNode.node_type,
        DecisionNode.content,
        DecisionNode.action_label,
        DecisionNode.tool_name,
        DecisionNode.subjects,
        DecisionNode.evidence_summary,
        DecisionNode.outcome_short,
        DecisionNode.decision_index,
        DecisionNode.embedding,
        Document.id.label("doc_id"),
        Document.skill_name,
        Document.subjects.label("doc_subjects"),
        Document.timestamp,
        Document.aftermath_status,
        Document.correction_text,
    ).join(
        Document,
        DecisionNode.document_id == Document.id,
    )

    if skill_name:
        stmt = stmt.where(Document.skill_name == skill_name)
    if action_label:
        stmt = stmt.where(DecisionNode.action_label == action_label)
    if aftermath_filter:
        stmt = stmt.where(Document.aftermath_status == aftermath_filter)

    stmt = stmt.order_by(Document.timestamp.desc()).limit(limit * 5)

    result = await session.execute(stmt)
    rows = result.fetchall()

    # Apply subjects filter (Python-level, JSON compat)
    if subjects:
        rows = [r for r in rows if _subjects_match(r[12], subjects)]

    use_vector = search_mode in ("hybrid", "vector")
    use_bm25 = search_mode in ("hybrid", "bm25")

    # Vector-only or BM25-only fast path
    if not use_bm25:
        # Vector only
        scored = []
        for row in rows:
            score = 0.0
            if query_embedding and row[9]:
                score = cosine_similarity(query_embedding, row[9])
            item = _build_item(row)
            item["rrf_score"] = round(score, 4)
            scored.append(item)
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored[:limit]

    if not use_vector:
        # BM25 only
        scored = []
        for row in rows:
            score = bm25_score(
                query_text,
                row[2],
                row[3] or "",
                subjects=_extract_subject_names(row[5]),
                evidence_summary=row[6] or "",
            )
            item = _build_item(row)
            item["rrf_score"] = round(score, 4)
            scored.append(item)
        scored.sort(
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        return scored[:limit]

    # Hybrid: BM25 + vector + RRF
    vector_list: list[dict] = []
    bm25_list: list[dict] = []

    for row in rows:
        item = _build_item(row)

        # Vector score
        vscore = 0.0
        if query_embedding and row[9]:
            vscore = cosine_similarity(query_embedding, row[9])
        v_item = item.copy()
        v_item["_score"] = vscore
        vector_list.append(v_item)

        # BM25 score
        bscore = bm25_score(
            query_text,
            row[2],
            row[3] or "",
            subjects=_extract_subject_names(row[5]),
            evidence_summary=row[6] or "",
        )
        b_item = item.copy()
        b_item["_score"] = bscore
        bm25_list.append(b_item)

    # Sort each list by its own score descending
    vector_list.sort(key=lambda x: x["_score"], reverse=True)
    bm25_list.sort(key=lambda x: x["_score"], reverse=True)

    # RRF merge
    merged = reciprocal_rank_fusion(vector_list, bm25_list, k=60)
    return merged[:limit]
