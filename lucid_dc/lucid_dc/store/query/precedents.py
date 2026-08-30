"""Layers 2+1 — Structured precedent query with aftermath.

H3:  Two-embedding weighted merge for precedents.
H4:  Candidates fetched without timestamp ordering.
H9:  Chain walk uses iterative LED_TO traversal.
H10: _find_corrected_outcome scoped to the node's doc.
H11: Subject filtering on decision_nodes.subjects JSON.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lucid_dc.shared.embedder import Embedder
from lucid_dc.shared.search import bm25_score, cosine_similarity
from lucid_dc.store.models import (
    DecisionNode,
    Document,
    Edge,
    Evidence,
)


async def query_precedents(
    session: AsyncSession,
    description: str,
    subjects: list[str] | None = None,
    evidence: str | None = None,
    limit: int = 5,
    embedder: Embedder | None = None,
    actor_id: str | None = None,
) -> dict:
    """Find relevant precedents with aftermath."""
    # H3: Embed description and evidence separately
    desc_embedding = None
    ev_embedding = None
    if embedder:
        desc_embedding = await embedder.embed(description)
        if evidence:
            ev_embedding = await embedder.embed(evidence)

    # Step 2: Find matching decision nodes
    candidates = await _find_candidates(
        session,
        description,
        subjects,
        limit * 5,  # H4: over-fetch, scoring handles rank
    )

    # Feature 3: Subject similarity boost via ABOUT edges.
    # Pre-compute a set of "related subjects" that share
    # corrections with the query subjects.
    related_subjects: dict[str, float] = {}
    if subjects:
        related_subjects = await _find_related_subjects(
            session,
            subjects,
        )

    # Step 3: Score and rank
    scored = _score_candidates(
        candidates,
        subjects,
        desc_embedding,
        ev_embedding,
        actor_id=actor_id,
        related_subjects=related_subjects,
        description=description,
    )

    # Step 4: Partition into corrected and non-corrected
    # BEFORE applying the limit. This ensures corrections
    # aren't pushed out by volume of accepted runs (e.g.
    # S17: early corrections followed by 16 accepted runs).
    corrected_candidates = [
        c
        for c in scored
        if c.get("document_metadata", {}).get(
            "aftermath_status",
        )
        == "corrected"
    ]
    other_candidates = [
        c
        for c in scored
        if c.get("document_metadata", {}).get(
            "aftermath_status",
        )
        != "corrected"
    ]

    # Dedup each pool by document_id separately.
    # Multiple nodes from the same doc carry the same
    # correction/status — keep the highest-scored node
    # per doc, but fetch ALL nodes from that doc and
    # concatenate their content so the full document
    # context is available for raw injection.
    corrected_deduped = await _dedup_by_doc(
        session,
        corrected_candidates,
    )
    other_deduped = await _dedup_by_doc(
        session,
        other_candidates,
    )

    # Merge: corrected first, then fill remaining
    # slots with non-corrected. Corrections always
    # have representation.
    deduped = corrected_deduped[:limit]
    remaining = limit - len(deduped)
    if remaining > 0:
        deduped.extend(other_deduped[:remaining])

    # Collect corrected doc_ids for outcome lookup
    corrected_doc_ids: set[str] = set()
    for c in deduped:
        doc = c.get("document_metadata", {})
        if doc.get("aftermath_status") == "corrected":
            doc_id = doc.get("id", "")
            if doc_id:
                corrected_doc_ids.add(doc_id)

    # One lookup per unique corrected document
    corrected_outcomes: dict[str, str | None] = {}
    for doc_id in corrected_doc_ids:
        outcome = await _find_corrected_outcome(
            session,
            doc_id,
        )
        if not outcome:
            outcome = await _find_correction_doc_outcome(
                session,
                doc_id,
            )
        corrected_outcomes[doc_id] = outcome

    # Resolve correction chains — if a corrected doc
    # was itself superseded by a later correction in
    # the same conversation, mark it as superseded
    # and surface the latest correction instead.
    chain_resolved = await _resolve_correction_chains(
        session,
        deduped,
        corrected_outcomes,
    )

    return {
        "decision_description": description,
        "subjects": subjects or [],
        "results": chain_resolved,
    }


async def _find_candidates(
    session: AsyncSession,
    description: str,
    subjects: list[str] | None,
    limit: int,
) -> list[dict]:
    """Find candidate decision nodes via SQL query.

    H4:  No timestamp ordering; scoring handles ranking.
    H11: When subjects are provided, filter nodes
    whose subjects JSON field contains any of the
    provided subject IDs.
    """
    stmt = (
        select(
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
            DecisionNode.accumulated_embedding,
            Document.id.label("doc_id"),
            Document.skill_name,
            Document.timestamp,
            Document.aftermath_status,
            Document.correction_text,
            Document.correction_needs_context,
            Document.actor_id,
        )
        .join(
            Document,
            DecisionNode.document_id == Document.id,
        )
        .limit(limit)
    )

    result = await session.execute(stmt)
    rows = result.fetchall()

    candidates = []
    for row in rows:
        node_subjects = row[5] or []

        # H11: Filter by subjects if provided.
        # Check if any provided subject is in the
        # node's subjects JSON array.
        if subjects and not _subjects_match(
            node_subjects,
            subjects,
        ):
            continue

        candidates.append(
            {
                "node_id": row[0],
                "node_type": row[1],
                "content": row[2],
                "action_label": row[3],
                "tool_name": row[4],
                "subjects": node_subjects,
                "evidence_summary": row[6] or "",
                "outcome_short": row[7] or "",
                "decision_index": row[8],
                "embedding": row[9],
                "accumulated_embedding": row[10],
                "document_metadata": {
                    "id": row[11],
                    "skill_name": row[12],
                    "timestamp": (row[13].isoformat() if row[13] else None),
                    "aftermath_status": row[14],
                    "correction_text": row[15],
                    "correction_needs_context": (row[16] or False),
                    "actor_id": row[17],
                },
            }
        )

    return candidates


def _subjects_match(
    node_subjects: list,
    query_subjects: list[str],
) -> bool:
    """Check if any query subject is in node subjects.

    H11: Handles both plain string lists and dicts
    with 'id' keys stored as JSON.
    """
    node_ids: set[str] = set()
    for s in node_subjects:
        if isinstance(s, str):
            node_ids.add(s)
        elif isinstance(s, dict) and "id" in s:
            node_ids.add(s["id"])
    return bool(node_ids & set(query_subjects))


def _extract_node_subjects(candidate: dict) -> set[str]:
    """Extract subject IDs from a candidate node."""
    node_subjects = candidate.get("subjects", [])
    node_ids: set[str] = set()
    if isinstance(node_subjects, list):
        for s in node_subjects:
            if isinstance(s, str):
                node_ids.add(s)
            elif isinstance(s, dict):
                node_ids.add(s.get("id", ""))
    return node_ids


def _score_candidates(
    candidates: list[dict],
    subjects: list[str] | None,
    desc_embedding: list[float] | None,
    ev_embedding: list[float] | None = None,
    actor_id: str | None = None,
    related_subjects: dict[str, float] | None = None,
    description: str = "",
) -> list[dict]:
    """Score candidates by relevance.

    Uses RRF to combine embedding similarity and BM25
    (different scales), then multiplicative boosts for
    subject/actor/aftermath signals.
    """
    if not candidates:
        return candidates

    # -- Compute raw signals per candidate --
    for candidate in candidates:
        node_emb = candidate.get("embedding")
        accum_emb = candidate.get("accumulated_embedding")

        # Score A: node embedding (specific data match)
        emb_score_a = 0.0
        if node_emb and desc_embedding:
            desc_sim = cosine_similarity(
                desc_embedding,
                node_emb,
            )
            if ev_embedding:
                ev_sim = cosine_similarity(
                    ev_embedding,
                    node_emb,
                )
                emb_score_a = 0.6 * desc_sim + 0.4 * ev_sim
            else:
                emb_score_a = desc_sim

        # Score B: accumulated embedding (stage match)
        emb_score_b = 0.0
        if accum_emb and desc_embedding:
            desc_sim_b = cosine_similarity(
                desc_embedding,
                accum_emb,
            )
            if ev_embedding:
                ev_sim_b = cosine_similarity(
                    ev_embedding,
                    accum_emb,
                )
                emb_score_b = 0.6 * desc_sim_b + 0.4 * ev_sim_b
            else:
                emb_score_b = desc_sim_b

        # Combined: node content weighted higher than
        # accumulated context — prioritizes "what happened"
        # over "where in the workflow"
        candidate["_emb_score"] = 0.7 * emb_score_a + 0.3 * emb_score_b

        # BM25 on correction text
        doc_meta = candidate.get("document_metadata", {})
        correction = doc_meta.get("correction_text") or ""
        bm25_val = 0.0
        if correction and description:
            bm25_val = bm25_score(description, correction)
        candidate["_bm25_score"] = bm25_val

    # -- Rank by each signal independently --
    by_emb = sorted(
        candidates,
        key=lambda c: c.get("_emb_score", 0),
        reverse=True,
    )
    by_bm25 = sorted(
        candidates,
        key=lambda c: c.get("_bm25_score", 0),
        reverse=True,
    )

    # -- RRF: combine embedding and BM25 ranks --
    k = 60
    emb_rank = {id(c): i for i, c in enumerate(by_emb)}
    bm25_rank = {id(c): i for i, c in enumerate(by_bm25)}

    for candidate in candidates:
        cid = id(candidate)
        base_rrf = 1.0 / (k + emb_rank.get(cid, len(candidates))) + 1.0 / (
            k + bm25_rank.get(cid, len(candidates))
        )

        # -- Multiplicative boosts --
        boost = 1.0
        doc_meta = candidate.get("document_metadata", {})
        node_ids = _extract_node_subjects(candidate)

        # Subject overlap
        if subjects:
            overlap = set(subjects) & node_ids
            boost += len(overlap) * 0.3

        # Subject similarity via ABOUT edges
        if related_subjects:
            for nid in node_ids:
                if nid in related_subjects:
                    boost += min(
                        related_subjects[nid] * 0.15,
                        0.15,
                    )

        # Aftermath boost
        if doc_meta.get("aftermath_status") == "corrected":
            boost += 0.2

        # Actor boost
        if actor_id and doc_meta.get("actor_id") == actor_id:
            boost += 0.3

        candidate["score"] = base_rrf * boost

        # Clean up temp fields
        candidate.pop("_emb_score", None)
        candidate.pop("_bm25_score", None)

    candidates.sort(
        key=lambda c: c.get("score", 0),
        reverse=True,
    )
    return candidates


async def _compute_aftermath(
    session: AsyncSession,
    node_id: str,
) -> dict:
    """Compute aftermath for a node via chain walk.

    H9: Uses iterative LED_TO traversal instead of
    just reading stored aftermath status. Walks
    forward from the node to find the first human
    response, then checks if it LED_TO another
    agent response (correction) or ended (acceptance).
    """
    # Get the node's document info
    stmt = (
        select(
            DecisionNode.document_id,
            Document.aftermath_status,
            Document.correction_text,
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
        return {"status": "pending"}

    doc_id = row[0]
    status = row[1] or "pending"
    correction = row[2]

    # H9: Walk LED_TO forward from the document's
    # last node to find human response.
    corrected_outcome = None
    if status == "corrected" and correction:
        corrected_outcome = await _find_corrected_outcome(
            session,
            doc_id,
        )
        # Fallback: if chain walk found nothing,
        # look for the next document in the same
        # conversation and use its outcome_summary.
        if not corrected_outcome:
            corrected_outcome = await _find_correction_doc_outcome(
                session,
                doc_id,
            )

    return {
        "status": status,
        "correction": correction,
        "corrected_outcome": corrected_outcome,
    }


async def _dedup_by_doc(
    session: AsyncSession,
    candidates: list[dict],
) -> list[dict]:
    """Dedup candidates by document_id.

    Keeps the highest-scored node per document, fetches
    full document content for raw injection.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for candidate in candidates:
        doc = candidate.get("document_metadata", {})
        doc_id = doc.get("id", "")
        if doc_id in seen:
            continue
        seen.add(doc_id)

        full_content = await _get_document_content(
            session,
            doc_id,
        )
        candidate["content"] = full_content
        result.append(candidate)
    return result


async def _get_document_content(
    session: AsyncSession,
    document_id: str,
) -> str:
    """Get full document content from all its nodes.

    Concatenates content from all nodes in decision
    order — tool calls, reasoning, assistant output.
    This gives the full document context for raw
    injection, matching what doc_document provides.
    """
    result = await session.execute(
        select(DecisionNode.content)
        .where(DecisionNode.document_id == document_id)
        .order_by(DecisionNode.decision_index.asc())
    )
    parts = [row[0] for row in result.fetchall() if row[0]]
    return "\n\n".join(parts)


async def _find_corrected_outcome(
    session: AsyncSession,
    document_id: str,
) -> str | None:
    """Walk forward from a doc's last node via
    FOLLOWED_BY to find the corrected outcome.

    H10: Scoped to the specific document's last node,
    not a global unscoped query.
    H9: Uses iterative queries for SQLite compat.
    """
    # Get last node of the document
    last_node_result = await session.execute(
        select(DecisionNode.id)
        .where(DecisionNode.document_id == document_id)
        .order_by(DecisionNode.decision_index.desc())
        .limit(1)
    )
    last_node_id = last_node_result.scalar_one_or_none()
    if not last_node_id:
        return None

    # Follow FOLLOWED_BY from this document's last
    # node to find the correction chain's first node.
    fb_result = await session.execute(
        select(Edge.to_id).where(
            Edge.edge_type == "FOLLOWED_BY",
            Edge.from_id == last_node_id,
        )
    )
    correction_node_id = fb_result.scalar_one_or_none()
    if not correction_node_id:
        return None

    # Walk LED_TO forward from the correction chain's
    # first node to find the last node (the outcome).
    current_id = correction_node_id
    visited: set[str] = {current_id}
    for _ in range(20):  # safety bound
        next_result = await session.execute(
            select(Edge.to_id).where(
                Edge.edge_type == "LED_TO",
                Edge.from_id == current_id,
            )
        )
        next_id = next_result.scalar_one_or_none()
        if not next_id or next_id in visited:
            break
        visited.add(next_id)
        current_id = next_id

    # Get the outcome_short from the final node
    outcome_result = await session.execute(
        select(DecisionNode.outcome_short).where(DecisionNode.id == current_id)
    )
    outcome = outcome_result.scalar_one_or_none()
    return outcome


async def _find_correction_doc_outcome(
    session: AsyncSession,
    original_doc_id: str,
) -> str | None:
    """Fallback: find the correction document's outcome.

    When the FOLLOWED_BY chain walk fails, find the next
    document in the same conversation (the correction)
    and return its outcome_summary.
    """
    # Get the original document's conversation_id
    orig = await session.execute(
        select(Document.conversation_id).where(
            Document.id == original_doc_id,
        )
    )
    conv_id = orig.scalar_one_or_none()
    if not conv_id:
        return None

    # Find the next document in that conversation
    # (the correction doc, captured after the original)
    correction_doc = await session.execute(
        select(Document.outcome_summary)
        .where(
            Document.conversation_id == conv_id,
            Document.id != original_doc_id,
        )
        .order_by(Document.timestamp.desc())
        .limit(1)
    )
    return correction_doc.scalar_one_or_none()


async def _resolve_correction_chains(
    session: AsyncSession,
    candidates: list[dict],
    corrected_outcomes: dict[str, str | None],
) -> list[dict]:
    """Group chain-linked corrections together.

    When multiple corrections are linked by parent_doc_id,
    group them as a numbered chain (latest first). The
    agent sees the chain structure and decides for itself
    whether corrections conflict (follow latest) or are
    additive (apply all).

    Corrections NOT in a chain are returned as regular
    [CORRECTED] entries.
    """
    # Build result dicts from candidates
    result_dicts: list[dict] = []
    doc_id_to_idx: dict[str, int] = {}

    for candidate in candidates:
        doc = candidate.get("document_metadata", {})
        doc_id = doc.get("id", "")
        status = doc.get("aftermath_status") or "pending"
        correction = doc.get("correction_text")

        result_dict = {
            "action_label": candidate.get(
                "action_label",
            ),
            "content": candidate.get("content", ""),
            "evidence_summary": candidate.get(
                "evidence_summary",
                "",
            ),
            "outcome_short": candidate.get(
                "outcome_short",
                "",
            ),
            "aftermath": status,
            "correction": correction,
            "corrected_outcome": (corrected_outcomes.get(doc_id)),
            "node_id": candidate["node_id"],
            "score": candidate.get("score", 0.0),
            "document_metadata": doc,
        }
        doc_id_to_idx[doc_id] = len(result_dicts)
        result_dicts.append(result_dict)

    # Find chain links among corrected results.
    # A chain exists when doc A's parent_doc_id points
    # to doc B, and both are in our result set.
    corrected_ids = {
        doc.get("document_metadata", {}).get("id")
        for doc in result_dicts
        if doc.get("aftermath") == "corrected"
    }

    # Build parent→child map from DB for corrected docs
    child_of: dict[str, str] = {}  # child_id → parent_id
    for doc_id in corrected_ids:
        if not doc_id:
            continue
        result = await session.execute(
            select(Document.parent_doc_id).where(
                Document.id == doc_id,
            )
        )
        parent_id = result.scalar_one_or_none()
        if parent_id and parent_id in corrected_ids:
            child_of[doc_id] = parent_id

    # Build chains: walk from each root to its leaves
    # A root is a corrected doc whose doc_id is NOT a
    # child of another corrected doc in the result set.
    parent_to_children: dict[str, list[str]] = {}
    for child_id, parent_id in child_of.items():
        parent_to_children.setdefault(
            parent_id,
            [],
        ).append(child_id)

    roots = corrected_ids - set(child_of.keys())
    in_chain: set[str] = set()
    chains: list[list[dict]] = []

    for root_id in roots:
        if not root_id:
            continue
        # Walk forward from root
        chain_ids: list[str] = [root_id]
        current = root_id
        visited: set[str] = {current}
        while current in parent_to_children:
            children = parent_to_children[current]
            # Pick first child (chains are linear)
            next_id = children[0]
            if next_id in visited:
                break
            chain_ids.append(next_id)
            visited.add(next_id)
            current = next_id

        if len(chain_ids) < 2:
            # Not a chain, just a single correction
            continue

        # Build chain: chronological order (oldest first)
        chain: list[dict] = []
        for cid in chain_ids:
            if cid in doc_id_to_idx:
                chain.append(result_dicts[doc_id_to_idx[cid]])
                in_chain.add(cid)
        if len(chain) >= 2:
            chains.append(chain)

    # Build final results:
    # 1. Chain groups (with chain metadata)
    # 2. Standalone corrected (not in any chain)
    # 3. Other (accepted/pending)
    chain_results: list[dict] = []
    for chain in chains:
        # Mark each result in the chain with position
        for i, item in enumerate(chain):
            item["chain_position"] = i + 1
            item["chain_length"] = len(chain)
            item["is_latest"] = i == len(chain) - 1
        chain_results.extend(chain)

    standalone: list[dict] = []
    other: list[dict] = []
    for r in result_dicts:
        doc_id = r.get("document_metadata", {}).get("id")
        if doc_id in in_chain:
            continue  # already in chain_results
        if r.get("aftermath") == "corrected":
            standalone.append(r)
        else:
            other.append(r)

    return chain_results + standalone + other


# ── Feature 2: Evidence-based retrieval ──────────────


async def query_evidence_corrections(
    session: AsyncSession,
    tool_result: str,
    skill_name: str,
    limit: int = 5,
    embedder: Embedder | None = None,
    subjects: list[str] | None = None,
) -> list[dict]:
    """Find corrections whose evidence matches tool result.

    Follows INFORMED_BY edges from DecisionNode → Evidence
    to find nodes that consumed similar data. Filters to
    corrected documents and ranks via hybrid retrieval:
    BM25 over Evidence.content combined with cosine
    similarity over the DecisionNode embedding (which
    represents the situation in which the evidence was
    consumed). The two ranks are merged with reciprocal
    rank fusion (RRF). When `subjects` is provided, a
    +0.2 boost per overlapping subject is added to the
    fused score to match the Stage 1 boost convention.

    Returns:
        List of dicts with correction, evidence_content,
        and score, sorted by hybrid relevance.
    """
    stmt = (
        select(
            Evidence.id,
            Evidence.content,
            Evidence.source,
            DecisionNode.id.label("node_id"),
            DecisionNode.content.label("node_content"),
            Document.id.label("doc_id"),
            Document.correction_text,
            Document.aftermath_status,
            DecisionNode.embedding,
            Document.subjects,
        )
        .join(
            Edge,
            (Edge.edge_type == "INFORMED_BY") & (Edge.to_id == Evidence.id),
        )
        .join(
            DecisionNode,
            DecisionNode.id == Edge.from_id,
        )
        .join(
            Document,
            DecisionNode.document_id == Document.id,
        )
        .where(
            Document.aftermath_status == "corrected",
            Document.skill_name == skill_name,
        )
    )

    result = await session.execute(stmt)
    rows = result.fetchall()

    # Embed the tool result once for the semantic side
    tool_embedding: list[float] | None = None
    if embedder:
        try:
            tool_embedding = await embedder.embed(tool_result)
        except Exception:
            tool_embedding = None

    query_subjects = set(subjects or [])

    candidates: list[dict] = []
    for row in rows:
        evidence_content = row[1] or ""
        correction = row[6] or ""
        if not correction:
            continue
        node_embedding = row[8]
        doc_subjects = row[9] or []
        if isinstance(doc_subjects, dict):
            doc_subjects = list(doc_subjects.values())
        bm25 = bm25_score(tool_result, evidence_content)
        cos = 0.0
        if tool_embedding is not None and node_embedding is not None:
            cos = cosine_similarity(tool_embedding, node_embedding)
        subj_overlap = 0
        if query_subjects:
            doc_subj_ids = {str(s.get("id") if isinstance(s, dict) else s) for s in doc_subjects}
            subj_overlap = len(query_subjects & doc_subj_ids)
        if bm25 <= 0 and cos <= 0 and subj_overlap == 0:
            continue
        candidates.append(
            {
                "evidence_id": row[0],
                "evidence_content": (evidence_content[:300]),
                "evidence_source": row[2],
                "node_id": row[3],
                "content": row[4] or "",
                "doc_id": row[5],
                "correction": correction,
                "aftermath": row[7],
                "bm25": bm25,
                "cos": cos,
                "subj_overlap": subj_overlap,
            }
        )

    if not candidates:
        return []

    # Reciprocal rank fusion of BM25 and cosine, plus a
    # subject overlap boost (+0.2 per overlapping subject)
    # to match the Stage 1 boost convention.
    bm25_ranked = sorted(
        candidates,
        key=lambda x: x["bm25"],
        reverse=True,
    )
    cos_ranked = sorted(
        candidates,
        key=lambda x: x["cos"],
        reverse=True,
    )
    bm25_rank = {id(c): i + 1 for i, c in enumerate(bm25_ranked)}
    cos_rank = {id(c): i + 1 for i, c in enumerate(cos_ranked)}
    k = 60
    for c in candidates:
        c["score"] = (
            1.0 / (k + bm25_rank[id(c)]) + 1.0 / (k + cos_rank[id(c)]) + 0.2 * c["subj_overlap"]
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )
    return candidates[:limit]


# ── Feature 3: Subject similarity via ABOUT edges ────


async def _find_related_subjects(
    session: AsyncSession,
    query_subjects: list[str],
) -> dict[str, float]:
    """Find subjects that share corrections with query.

    Uses ABOUT edges to find documents that touch the
    query subjects, then finds OTHER subjects on those
    same corrected documents. Returns a mapping of
    subject_id → similarity score (0-1).

    Single batch query.
    """
    if not query_subjects:
        return {}

    # Find corrected documents connected to query
    # subjects via ABOUT edges.
    # ABOUT edges: DecisionNode → Subject
    # We need: Subject ← ABOUT ← Node → Document
    stmt = (
        select(
            Edge.from_id.label("node_id"),
            DecisionNode.document_id,
        )
        .join(
            DecisionNode,
            DecisionNode.id == Edge.from_id,
        )
        .join(
            Document,
            DecisionNode.document_id == Document.id,
        )
        .where(
            Edge.edge_type == "ABOUT",
            Edge.to_id.in_(query_subjects),
            Document.aftermath_status == "corrected",
        )
    )

    result = await session.execute(stmt)
    corrected_doc_ids = {row[1] for row in result.fetchall()}

    if not corrected_doc_ids:
        return {}

    # Find OTHER subjects on those corrected documents.
    stmt2 = (
        select(
            Edge.to_id.label("subject_id"),
        )
        .join(
            DecisionNode,
            DecisionNode.id == Edge.from_id,
        )
        .where(
            Edge.edge_type == "ABOUT",
            DecisionNode.document_id.in_(corrected_doc_ids),
            ~Edge.to_id.in_(query_subjects),
        )
    )

    result2 = await session.execute(stmt2)
    subject_counts: dict[str, int] = {}
    for row in result2.fetchall():
        sid = row[0]
        subject_counts[sid] = subject_counts.get(sid, 0) + 1

    if not subject_counts:
        return {}

    # Normalize to 0-1 range
    max_count = max(subject_counts.values())
    return {sid: count / max_count for sid, count in subject_counts.items()}
