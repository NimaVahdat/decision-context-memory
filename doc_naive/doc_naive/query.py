"""Doc query — search + global readout.

Precedents, chain, why are limited in this approach
(no edges, no singletons). Precedents falls back to
search. Chain/why raise NotImplementedError.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import distinct, func, select

from doc_naive.config import OpenAIConfig
from doc_naive.engine import DocEngine
from doc_naive.interface import QueryStrategy
from doc_naive.models import (
    Chunk,
    DocDocument,
)
from doc_naive.shared.embedder import Embedder
from doc_naive.shared.models import ChainNode, WhyResult
from doc_naive.shared.search import (
    bm25_score,
    cosine_similarity,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


class DocQuery(QueryStrategy):
    """Query strategy for this approach."""

    def __init__(
        self,
        db_url: str,
        openai_config: OpenAIConfig,
        result_mode: str = "chunk",
        naive: bool = False,
    ) -> None:
        self._engine = DocEngine(db_url)
        self._openai_config = openai_config
        self._embedder: Embedder | None = None
        self._result_mode = result_mode
        self._naive = naive

    @property
    def format_mode(self) -> str:
        return "naive" if self._naive else "raw"

    async def initialize(self) -> None:
        await self._engine.create_tables()
        if self._openai_config.api_key:
            self._embedder = Embedder(
                api_key=self._openai_config.api_key,
                model=self._openai_config.embedding_model,
            )

    async def shutdown(self) -> None:
        await self._engine.dispose()

    async def global_readout(
        self,
        skill_name: str,
        min_runs: int,
        actor_id: str | None = None,
    ) -> dict:
        """Skill-level aggregation over chunks."""
        async with self._engine.session() as session:
            stmt = (
                select(
                    Chunk.action_label,
                    func.count().label("runs"),
                    func.count()
                    .filter(DocDocument.aftermath_status == "corrected")
                    .label("corrections"),
                    func.count(distinct(Chunk.outcome_short)).label("outcome_variety"),
                )
                .join(
                    DocDocument,
                    Chunk.document_id == DocDocument.id,
                )
                .where(DocDocument.skill_name == skill_name)
                .group_by(Chunk.action_label)
                .having(func.count() >= min_runs)
                .order_by(func.count().filter(DocDocument.aftermath_status == "corrected").desc())
            )
            result = await session.execute(stmt)
            rows = result.fetchall()

            total_stmt = select(func.count(distinct(DocDocument.id))).where(
                DocDocument.skill_name == skill_name
            )
            total_result = await session.execute(total_stmt)
            total_runs = total_result.scalar() or 0

            actions = []
            for row in rows:
                (
                    action_label,
                    runs,
                    corrections,
                    outcome_variety,
                ) = row
                is_settled = runs >= 10 and corrections == 0 and outcome_variety <= 3
                correction_rate = corrections / runs if runs > 0 else 0.0

                patterns_stmt = (
                    select(distinct(DocDocument.correction_text))
                    .join(
                        Chunk,
                        Chunk.document_id == DocDocument.id,
                    )
                    .where(
                        DocDocument.skill_name == skill_name,
                        Chunk.action_label == action_label,
                        DocDocument.correction_text.isnot(None),
                    )
                )
                patterns_result = await session.execute(patterns_stmt)
                correction_patterns = [p[0] for p in patterns_result.fetchall() if p[0]]

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
        """BM25 + vector hybrid search over chunks.

        result_mode:
          "chunk" — return individual matched chunks
          "document" — return full documents (all chunks)
            for each matched run, ranked by best chunk
        """
        # Step 1: Find matched chunks (always chunk-level)
        chunk_results = await self._search_chunks(
            query_text=query_text,
            subjects=subjects,
            skill_name=skill_name,
            action_label=action_label,
            aftermath_filter=aftermath_filter,
            search_mode=search_mode,
            limit=limit if result_mode == "chunk" else limit * 3,
        )

        if result_mode == "chunk":
            return chunk_results[:limit]

        # Step 2: Document mode — expand to full documents
        return await self._expand_to_documents(chunk_results, limit)

    async def _search_chunks(
        self,
        query_text: str,
        subjects: list[str] | None,
        skill_name: str | None,
        action_label: str | None,
        aftermath_filter: str | None,
        search_mode: str,
        limit: int,
    ) -> list[dict]:
        """Core chunk-level search with BM25+vector+RRF."""
        query_embedding = None
        if self._embedder:
            query_embedding = await self._embedder.embed(query_text)

        async with self._engine.session() as session:
            stmt = select(
                Chunk.id,
                Chunk.content,
                Chunk.action_label,
                Chunk.tool_name,
                Chunk.evidence_summary,
                Chunk.outcome_short,
                Chunk.decision_index,
                Chunk.embedding,
                DocDocument.id.label("doc_id"),
                DocDocument.skill_name,
                DocDocument.subjects.label("doc_subjects"),
                DocDocument.timestamp,
                DocDocument.aftermath_status,
                DocDocument.correction_text,
                DocDocument.conversation_id,
                DocDocument.outcome_summary,
                DocDocument.correction_needs_context,
            ).join(
                DocDocument,
                Chunk.document_id == DocDocument.id,
            )

            if skill_name:
                stmt = stmt.where(DocDocument.skill_name == skill_name)
            if action_label:
                stmt = stmt.where(Chunk.action_label == action_label)
            if aftermath_filter:
                stmt = stmt.where(DocDocument.aftermath_status == aftermath_filter)

            stmt = stmt.order_by(DocDocument.timestamp.desc()).limit(limit * 5)

            result = await session.execute(stmt)
            rows = result.fetchall()

        if subjects:
            rows = [r for r in rows if _subjects_match(r[10], subjects)]

        use_vector = search_mode in ("hybrid", "vector")
        use_bm25 = search_mode in ("hybrid", "bm25")

        if not use_bm25:
            scored = []
            for row in rows:
                score = 0.0
                if query_embedding and row[7]:
                    score = cosine_similarity(query_embedding, row[7])
                item = _build_item(row)
                item["rrf_score"] = round(score, 4)
                scored.append(item)
            scored.sort(
                key=lambda x: x["rrf_score"],
                reverse=True,
            )
            return scored[:limit]

        if not use_vector:
            scored = []
            for row in rows:
                score = bm25_score(
                    query_text,
                    row[1],
                    row[2] or "",
                    evidence_summary=row[4] or "",
                )
                item = _build_item(row)
                item["rrf_score"] = round(score, 4)
                scored.append(item)
            scored.sort(
                key=lambda x: x["rrf_score"],
                reverse=True,
            )
            return scored[:limit]

        vector_list: list[dict] = []
        bm25_list: list[dict] = []

        for row in rows:
            item = _build_item(row)

            vscore = 0.0
            if query_embedding and row[7]:
                vscore = cosine_similarity(query_embedding, row[7])
            v_item = item.copy()
            v_item["_score"] = vscore
            vector_list.append(v_item)

            bscore = bm25_score(
                query_text,
                row[1],
                row[2] or "",
                evidence_summary=row[4] or "",
            )
            b_item = item.copy()
            b_item["_score"] = bscore
            bm25_list.append(b_item)

        vector_list.sort(key=lambda x: x["_score"], reverse=True)
        bm25_list.sort(key=lambda x: x["_score"], reverse=True)

        merged = reciprocal_rank_fusion(vector_list, bm25_list, k=60)
        return merged[:limit]

    async def _expand_to_documents(
        self,
        chunk_results: list[dict],
        limit: int,
    ) -> list[dict]:
        """Expand chunk results to full documents.

        Groups matched chunks by document, fetches all
        chunks for each doc, returns documents ranked
        by their best matching chunk's score.
        """
        # Deduplicate by document, keep best score
        seen_docs: dict[str, dict] = {}
        for chunk in chunk_results:
            doc_id = chunk.get("document", {}).get("id")
            if not doc_id:
                continue
            if doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    "doc_id": doc_id,
                    "best_score": chunk.get("rrf_score", 0),
                    "doc_meta": chunk.get("document", {}),
                }
            else:
                existing = seen_docs[doc_id]
                score = chunk.get("rrf_score", 0)
                if score > existing["best_score"]:
                    existing["best_score"] = score

        # Rank documents by best chunk score
        ranked_docs = sorted(
            seen_docs.values(),
            key=lambda d: d["best_score"],
            reverse=True,
        )[:limit]

        # Fetch all chunks for each matched document
        results = []
        async with self._engine.session() as session:
            for doc_info in ranked_docs:
                doc_id = doc_info["doc_id"]

                doc = await session.get(DocDocument, doc_id)
                if not doc:
                    continue

                chunks_result = await session.execute(
                    select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
                )
                chunks = chunks_result.scalars().all()

                results.append(
                    {
                        "document_id": doc_id,
                        "rrf_score": doc_info["best_score"],
                        "skill_name": doc.skill_name,
                        "actor_id": doc.actor_id,
                        "subjects": doc.subjects,
                        "aftermath_status": (doc.aftermath_status),
                        "correction_text": (doc.correction_text),
                        "outcome_summary": (doc.outcome_summary),
                        "conversation_id": (doc.conversation_id),
                        "correction_needs_context": (doc.correction_needs_context),
                        "timestamp": (doc.timestamp.isoformat() if doc.timestamp else None),
                        "chunks": [
                            {
                                "id": c.id,
                                "content": c.content,
                                "action_label": (c.action_label),
                                "tool_name": c.tool_name,
                                "evidence_summary": (c.evidence_summary or ""),
                                "outcome_short": (c.outcome_short or ""),
                                "chunk_index": (c.chunk_index),
                                "decision_index": (c.decision_index),
                            }
                            for c in chunks
                        ],
                    }
                )

        return results

    async def precedents(
        self,
        description: str,
        subjects: list[str] | None = None,
        evidence: str | None = None,
        limit: int = 5,
        actor_id: str | None = None,
    ) -> dict:
        """Precedents via search.

        Uses the configured result_mode so doc_document
        returns full documents (all chunks) while
        doc_chunk returns individual matched chunks.
        """
        results = await self.search(
            query_text=description,
            subjects=subjects,
            result_mode=self._result_mode,
            limit=limit,
        )

        precedent_results = []
        for r in results:
            if self._result_mode == "document":
                # Document mode: fields are top-level,
                # not nested under "document".
                chunks = r.get("chunks", [])
                first_chunk = chunks[0] if chunks else {}
                # Assemble full content from all chunks
                all_content = "\n\n".join(c.get("content", "") for c in chunks if c.get("content"))
                precedent_results.append(
                    {
                        "action_label": first_chunk.get(
                            "action_label",
                        ),
                        "evidence_summary": first_chunk.get(
                            "evidence_summary",
                            "",
                        ),
                        "outcome_short": r.get(
                            "outcome_summary",
                            "",
                        ),
                        "content": all_content,
                        "aftermath": r.get(
                            "aftermath_status",
                            "pending",
                        ),
                        "correction": r.get(
                            "correction_text",
                        ),
                        "corrected_outcome": None,
                        "node_id": r.get(
                            "document_id",
                            "",
                        ),
                        "document_metadata": {
                            "id": r.get("document_id"),
                            "skill_name": r.get(
                                "skill_name",
                            ),
                            "subjects": r.get("subjects"),
                            "aftermath_status": r.get(
                                "aftermath_status",
                            ),
                            "correction_text": r.get(
                                "correction_text",
                            ),
                            "conversation_id": r.get(
                                "conversation_id",
                            ),
                            "outcome_summary": r.get(
                                "outcome_summary",
                            ),
                            "correction_needs_context": r.get(
                                "correction_needs_context",
                                False,
                            ),
                        },
                    }
                )
            else:
                # Chunk mode: metadata nested under
                # "document" key.
                doc_meta = r.get("document", {})
                precedent_results.append(
                    {
                        "action_label": r.get(
                            "action_label",
                        ),
                        "evidence_summary": r.get(
                            "evidence_summary",
                            "",
                        ),
                        "outcome_short": r.get(
                            "outcome_short",
                            "",
                        ),
                        "content": r.get(
                            "content",
                            "",
                        ),
                        "aftermath": doc_meta.get(
                            "aftermath_status",
                            "pending",
                        ),
                        "correction": doc_meta.get(
                            "correction_text",
                        ),
                        "corrected_outcome": None,
                        "node_id": r.get("node_id", ""),
                        "document_metadata": doc_meta,
                    }
                )

        return {
            "decision_description": description,
            "subjects": subjects or [],
            "results": precedent_results,
        }

    async def chain(
        self,
        node_id: str,
        direction: str = "forward",
        max_depth: int = 10,
    ) -> list[ChainNode]:
        """Doc approach has no LED_TO edges."""
        async with self._engine.session() as session:
            chunk = await session.get(Chunk, node_id)
            if not chunk:
                return []

            stmt = (
                select(
                    Chunk.id,
                    Chunk.action_label,
                    Chunk.outcome_short,
                    DocDocument.actor_id,
                    DocDocument.actor_type,
                )
                .join(
                    DocDocument,
                    Chunk.document_id == DocDocument.id,
                )
                .where(Chunk.document_id == chunk.document_id)
                .order_by(Chunk.chunk_index)
            )
            result = await session.execute(stmt)
            rows = result.fetchall()

        return [
            ChainNode(
                id=row[0],
                action_label=row[1],
                outcome_short=row[2] or "",
                subjects=[],
                actor_id=row[3],
                actor_type=row[4],
                depth=i,
            )
            for i, row in enumerate(rows)
        ]

    async def why(self, node_id: str) -> WhyResult:
        """Doc approach: returns chunk content."""
        async with self._engine.session() as session:
            chunk = await session.get(Chunk, node_id)
            if not chunk:
                raise LookupError(f"Chunk {node_id} not found")

            return WhyResult(
                node_id=node_id,
                reason=chunk.content,
                cached=True,
            )

    async def get_document(self, document_id: str) -> dict:
        """Retrieve a full document with all chunks."""
        async with self._engine.session() as session:
            doc = await session.get(DocDocument, document_id)
            if not doc:
                raise LookupError(f"Document {document_id} not found")

            chunks_result = await session.execute(
                select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
            )
            chunks = chunks_result.scalars().all()

            return {
                "id": doc.id,
                "skill_name": doc.skill_name,
                "skill_version": doc.skill_version,
                "actor_id": doc.actor_id,
                "actor_type": doc.actor_type,
                "subjects": doc.subjects,
                "conversation_id": doc.conversation_id,
                "parent_doc_id": doc.parent_doc_id,
                "aftermath_status": (doc.aftermath_status),
                "correction_text": doc.correction_text,
                "outcome_summary": doc.outcome_summary,
                "message_history": doc.message_history,
                "timestamp": (doc.timestamp.isoformat() if doc.timestamp else None),
                "chunks": [
                    {
                        "id": c.id,
                        "node_type": "chunk",
                        "content": c.content,
                        "action_label": (c.action_label),
                        "tool_name": c.tool_name,
                        "subjects": [],
                        "evidence_summary": (c.evidence_summary),
                        "outcome_short": (c.outcome_short),
                        "chunk_index": c.chunk_index,
                        "decision_index": (c.decision_index),
                        "reason": None,
                        "evidence": None,
                        "edges": [],
                    }
                    for c in chunks
                ],
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
                select(DocDocument)
                .where(DocDocument.conversation_id == conversation_id)
                .order_by(DocDocument.timestamp)
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


def _subjects_match(
    doc_subjects: list | dict | None,
    filter_ids: list[str],
) -> bool:
    """Check if document subjects contain any filter ID.

    Subjects stored as JSON (list of dicts or strings).
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
    """Build a search result dict from a DB row.

    D1: Includes conversation_id and outcome_summary.
    """
    return {
        "node_id": row[0],
        "node_type": "chunk",
        "content": row[1],
        "action_label": row[2],
        "tool_name": row[3],
        "subjects": [],
        "evidence_summary": row[4] or "",
        "outcome_short": row[5] or "",
        "decision_index": row[6] or 0,
        "document": {
            "id": row[8],
            "skill_name": row[9],
            "subjects": row[10],
            "timestamp": (row[11].isoformat() if row[11] else None),
            "aftermath_status": row[12],
            "correction_text": row[13],
            "conversation_id": row[14],
            "outcome_summary": row[15],
            "correction_needs_context": row[16] or False,
        },
    }
