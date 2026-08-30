"""Doc capture pipeline — chunk + annotate + embed + store."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime

from openai import AsyncOpenAI
from sqlalchemy import select

from doc_naive.config import OpenAIConfig
from doc_naive.engine import DocEngine
from doc_naive.interface import CaptureStrategy
from doc_naive.models import (
    Chunk,
    Conversation,
    DocDocument,
)
from doc_naive.shared.aftermath import (
    classify_followup,
    classify_followup_deterministic,
)
from doc_naive.shared.embedder import Embedder

logger = logging.getLogger(__name__)


class DocCapture(CaptureStrategy):
    """Capture strategy for this approach."""

    def __init__(self, db_url: str, openai_config: OpenAIConfig) -> None:
        self._engine = DocEngine(db_url)
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

    async def capture(self, trace: dict) -> dict:
        """Run the doc capture pipeline."""
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
        skill_definition = trace.get("skill_definition", "")

        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now(UTC)

        subject_ids = [s["id"] if isinstance(s, dict) else s for s in subjects_raw]

        # Generate outcome_summary if not provided
        if not outcome_summary:
            outcome_summary = _generate_outcome_summary(skill_name, subject_ids, message_history)

        # Chunk by tool-call groups + annotate
        raw_chunks = _chunk_by_tool_groups(message_history)
        annotated = _annotate_chunks(
            raw_chunks,
            skill_definition=skill_definition,
        )

        # Compute embeddings
        if self._embedder:
            texts = [c["content"] for c in annotated]
            embeddings = await self._embedder.embed_batch(texts)
        else:
            embeddings = [None] * len(annotated)

        # Store document + chunks + conversation
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

            for i, ann in enumerate(annotated):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    content=ann["content"],
                    chunk_index=i,
                    embedding=embeddings[i],
                    action_label=ann["action_label"],
                    tool_name=ann["tool_name"],
                    evidence_summary=ann["evidence_summary"],
                    outcome_short=ann["outcome_short"],
                    decision_index=ann["decision_index"],
                )
                session.add(chunk)

            # D2: Create or update Conversation row
            if conversation_id:
                await _upsert_conversation(session, conversation_id, doc_id)

            # Lazy aftermath for previous doc
            aftermath_updated = None
            if conversation_id:
                aftermath_updated = await _compute_lazy_aftermath(
                    session,
                    doc_id,
                    conversation_id,
                    message_history,
                    self._openai_client,
                    new_skill=skill_name,
                    explicit_parent_id=parent_doc_id,
                )

        return {
            "document_id": doc_id,
            "chunks_created": len(annotated),
            "aftermath_updated": aftermath_updated,
            "timestamp": timestamp.isoformat(),
        }

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and its chunks.

        Returns the number of chunks deleted.
        """
        async with self._engine.session() as session:
            result = await session.execute(select(Chunk).where(Chunk.document_id == document_id))
            chunks = result.scalars().all()
            chunks_deleted = len(chunks)
            for chunk in chunks:
                await session.delete(chunk)

            doc = await session.get(DocDocument, document_id)
            if not doc:
                raise LookupError(f"Document {document_id} not found")
            await session.delete(doc)

        return chunks_deleted


# ── Chunking ─────────────────────────────────────────


def _split_at_semantic_breaks(
    content: str,
    target_tokens: int = 900,
) -> list[str]:
    """Split content at semantic boundaries.

    Estimates tokens as word count (rough 1:1 approximation).
    Splits at highest-scoring boundary, never inside fenced
    code blocks. Minimum chunk size: 100 tokens. Adds ~15%
    overlap between consecutive chunks.
    """
    words = content.split()
    if len(words) <= target_tokens:
        return [content]

    lines = content.split("\n")
    # Score each line boundary
    scored_boundaries: list[tuple[int, int]] = []
    in_code_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
        if in_code_block:
            continue
        if i == 0:
            continue

        score = 0
        if stripped.startswith("# "):
            score = 100
        elif stripped.startswith("## "):
            score = 90
        elif stripped.startswith("###"):
            score = 80
        elif stripped == "" and i > 0:
            prev = lines[i - 1].strip()
            if prev != "":
                score = 20

        if score > 0:
            scored_boundaries.append((i, score))

    if not scored_boundaries:
        return [content]

    scored_boundaries.sort(key=lambda x: x[1], reverse=True)

    # Greedily split at boundaries respecting target_tokens
    split_indices = _select_split_points(lines, scored_boundaries, target_tokens)

    if not split_indices:
        return [content]

    split_indices = sorted(set(split_indices))
    segments: list[str] = []
    prev = 0
    for idx in split_indices:
        seg = "\n".join(lines[prev:idx])
        if seg.strip():
            segments.append(seg)
        prev = idx
    tail = "\n".join(lines[prev:])
    if tail.strip():
        segments.append(tail)

    # Merge small trailing chunks (< 100 tokens)
    merged = _merge_small_chunks(segments, min_tokens=100)

    # Add 15% overlap
    overlapped = _add_overlap(merged, fraction=0.15)

    return overlapped


def _select_split_points(
    lines: list[str],
    scored_boundaries: list[tuple[int, int]],
    target_tokens: int,
) -> list[int]:
    """Greedily pick split points near target_tokens.

    D6: Scans lines, tracking word count. When the
    count exceeds target_tokens, picks the best
    boundary seen so far and resets the window.
    """
    boundary_set = {b[0]: b[1] for b in scored_boundaries}

    selected: list[int] = []
    word_count = 0
    best_boundary: int | None = None
    best_score = -1

    for i, line in enumerate(lines):
        word_count += len(line.split())

        if i in boundary_set and boundary_set[i] > best_score:
            best_boundary = i
            best_score = boundary_set[i]

        if word_count >= target_tokens and best_boundary is not None:
            selected.append(best_boundary)
            # Re-count words from the boundary to now
            word_count = sum(len(lines[j].split()) for j in range(best_boundary, i + 1))
            best_boundary = None
            best_score = -1

    return selected


def _merge_small_chunks(
    segments: list[str],
    min_tokens: int,
) -> list[str]:
    """Merge segments smaller than min_tokens with previous."""
    if not segments:
        return segments

    merged: list[str] = [segments[0]]
    for seg in segments[1:]:
        word_count = len(seg.split())
        if word_count < min_tokens and merged:
            merged[-1] = merged[-1] + "\n" + seg
        else:
            merged.append(seg)

    return merged


def _add_overlap(
    segments: list[str],
    fraction: float,
) -> list[str]:
    """Carry last ~fraction of previous chunk into next."""
    if len(segments) <= 1:
        return segments

    result = [segments[0]]
    for i in range(1, len(segments)):
        prev_words = segments[i - 1].split()
        overlap_size = max(1, int(len(prev_words) * fraction))
        overlap_text = " ".join(prev_words[-overlap_size:])
        result.append(overlap_text + "\n" + segments[i])

    return result


def _has_tool_call(messages: list[dict]) -> bool:
    """Check if a message list contains a tool_call."""
    return any(m.get("role") == "tool_call" for m in messages)


def _chunk_by_tool_groups(
    messages: list[dict],
) -> list[dict]:
    """Group messages into tool-call group chunks.

    Each chunk = assistant reasoning + tool_call + tool_result.
    Non-tool assistant messages become their own chunks.
    Non-tool chunks that are too long get split at semantic
    break points.
    """
    chunks: list[dict] = []
    buffer: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            continue
        buffer.append(msg)
        if role == "tool_result":
            chunks.append({"messages": list(buffer)})
            buffer = []

    if buffer:
        # Filter to only non-empty assistant messages
        has_content = any(
            m.get("role") == "assistant" and m.get("content", "").strip() for m in buffer
        )
        if has_content:
            chunks.append({"messages": list(buffer)})

    # Second pass: split non-tool chunks that are too long
    final_chunks: list[dict] = []
    for chunk in chunks:
        if _has_tool_call(chunk["messages"]):
            final_chunks.append(chunk)
            continue

        content = "\n".join(
            m.get("content", "") for m in chunk["messages"] if m.get("content", "").strip()
        )
        parts = _split_at_semantic_breaks(content)
        if len(parts) <= 1:
            final_chunks.append(chunk)
        else:
            for part in parts:
                final_chunks.append(
                    {
                        "messages": chunk["messages"],
                        "split_content": part,
                    }
                )

    return final_chunks


def _annotate_chunks(
    raw_chunks: list[dict],
    skill_definition: str = "",
) -> list[dict]:
    """Annotate each chunk with 5 deterministic fields.

    D4: outcome_short extracted from the last assistant
    message in the chunk, not from assembled content.
    """
    annotated = []
    for idx, chunk in enumerate(raw_chunks):
        msgs = chunk["messages"]
        split_content = chunk.get("split_content")

        # Build content from messages
        content_parts = []
        tool_name = None
        tool_result_text = ""

        for msg in msgs:
            role = msg.get("role", "")
            if role == "assistant":
                text = msg.get("content", "")
                if text.strip():
                    content_parts.append(text)
            elif role == "tool_call":
                tool_name = msg.get("name", "")
                args = msg.get("arguments", {})
                content_parts.append(f"[{tool_name}({args})]")
            elif role == "tool_result":
                tool_result_text = msg.get("content", "")
                truncated = tool_result_text[:200]
                content_parts.append(f"→ {truncated}")

        # Use split_content if this chunk was semantically
        # split, otherwise assemble from parts
        content = split_content if split_content else "\n".join(content_parts)

        # action_label: tool name or heuristic
        action_label = tool_name or _heuristic_label(
            content, idx, skill_definition=skill_definition
        )

        # evidence_summary: first 200 chars of tool result
        evidence_summary = tool_result_text[:200] if tool_result_text else ""

        # D4: outcome_short from last assistant message
        outcome_short = _extract_outcome_from_messages(msgs)

        annotated.append(
            {
                "content": content,
                "action_label": action_label,
                "tool_name": tool_name,
                "evidence_summary": evidence_summary,
                "outcome_short": outcome_short,
                "decision_index": idx,
            }
        )

    return annotated


def _heuristic_label(
    content: str,
    index: int,
    skill_definition: str = "",
) -> str:
    """Assign a heuristic label to non-tool chunks.

    If skill_definition is provided, extract keywords from it
    and match against content. Falls back to generic keywords.
    """
    lower = content.lower()

    if skill_definition:
        label = _match_skill_keywords(lower, skill_definition)
        if label:
            return label

    if any(w in lower for w in ["threshold", "evaluate", "assess"]):
        return "evaluation"
    if any(w in lower for w in ["format", "report", "summary"]):
        return "synthesis"
    if any(w in lower for w in ["plan", "decide", "choose"]):
        return "planning"
    return f"reasoning_{index}"


def _match_skill_keywords(
    content_lower: str,
    skill_definition: str,
) -> str | None:
    """Extract keywords from skill_definition, find matches.

    Split skill_definition into words, look for 2+ word
    phrases that appear in content, return as underscore label.
    """
    words = [w.lower().strip(".,;:!?()[]{}\"'") for w in skill_definition.split() if len(w) > 3]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    # Look for 2-word phrases from skill_definition in content
    matched: list[str] = []
    for i in range(len(unique) - 1):
        phrase = f"{unique[i]} {unique[i + 1]}"
        if phrase in content_lower:
            matched.append(phrase)
        if len(matched) >= 2:
            break

    if matched:
        return matched[0].replace(" ", "_")

    # Fall back to single keyword matches
    single_matches = [w for w in unique if w in content_lower]
    if len(single_matches) >= 2:
        return "_".join(single_matches[:2])

    return None


def _extract_outcome(content: str) -> str:
    """Extract the last sentence, truncated to 100 chars.

    H7: Splits on whitespace after punctuation so that
    terminal punctuation is preserved in each segment.
    """
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""
    last = sentences[-1]
    if len(last) > 100:
        return last[:97] + "..."
    return last


def _extract_outcome_from_messages(
    messages: list[dict],
) -> str:
    """Extract outcome from the last assistant message.

    D4: Uses the last assistant message in the chunk's message
    list rather than the assembled content string.
    """
    last_assistant_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = msg.get("content", "").strip()
            if text:
                last_assistant_text = text
                break

    if not last_assistant_text:
        return ""
    return _extract_outcome(last_assistant_text)


def _generate_outcome_summary(
    skill_name: str,
    subject_ids: list[str],
    messages: list[dict],
) -> str:
    last_assistant = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break
    subject_str = ", ".join(subject_ids) if subject_ids else ""
    return f"{skill_name} for {subject_str}: {last_assistant[:200]}"


async def _upsert_conversation(
    session,
    conversation_id: str,
    doc_id: str,
) -> None:
    """Create or update a Conversation row.

    Adds doc_id to the conversation's documents list
    and updates last_updated.
    """
    conv = await session.get(Conversation, conversation_id)
    if conv is None:
        conv = Conversation(
            id=conversation_id,
            documents=[doc_id],
        )
        session.add(conv)
    else:
        docs = list(conv.documents or [])
        if doc_id not in docs:
            docs.append(doc_id)
        conv.documents = docs
        conv.last_updated = datetime.now(UTC)


async def _compute_lazy_aftermath(
    session,
    new_doc_id: str,
    conversation_id: str,
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
    """
    if explicit_parent_id:
        result = await session.execute(
            select(DocDocument).where(
                DocDocument.id == explicit_parent_id,
            )
        )
    else:
        result = await session.execute(
            select(DocDocument)
            .where(
                DocDocument.conversation_id == conversation_id,
                DocDocument.id != new_doc_id,
                DocDocument.aftermath_status == "pending",
            )
            .order_by(DocDocument.timestamp.desc())
            .limit(1)
        )
    prev_doc = result.scalar_one_or_none()
    if not prev_doc:
        return None

    human_message = ""
    for msg in new_messages:
        if msg.get("role") == "user":
            human_message = msg.get("content", "")
            break

    if not human_message:
        return None

    # Try deterministic classification first, fall back to LLM.
    classification = classify_followup_deterministic(
        human_message,
    )
    needs_context: bool | None = None

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

        # Determine if correction needs context.
        if needs_context is not None:
            prev_doc.correction_needs_context = needs_context
        else:
            from doc_naive.shared.ambiguity import (
                correction_needs_context,
            )

            prev_doc.correction_needs_context = correction_needs_context(human_message)
    else:
        prev_doc.aftermath_status = "accepted"

    return prev_doc.id
