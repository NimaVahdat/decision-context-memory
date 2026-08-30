"""Consolidation: LLM analyzes traces and produces directives + insights.

One LLM call reads all captured traces and produces:
1. Tool-calling directives ("DO NOT call X") for failures
2. Knowledge insights (few-shot examples) for quality patterns

Uses Anthropic haiku.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lucid_dc.capture.models import (
    ConsolidationResult,
)
from lucid_dc.store.models import Document

logger = logging.getLogger(__name__)

TRACE_SUMMARY_LIMIT = 10


# -- Gather trace data ----------------------------------------


async def gather_trace_summaries(
    session: AsyncSession,
    skill_name: str,
    limit: int = TRACE_SUMMARY_LIMIT,
) -> list[dict]:
    """Build compact trace summaries for the LLM analyzer.

    Includes tool call details with result counts for
    quality comparison across traces.
    """
    stmt = (
        select(Document)
        .where(Document.skill_name == skill_name)
        .order_by(Document.timestamp.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    docs = list(result.scalars().all())

    summaries = []
    for doc in docs:
        subjects = []
        if doc.subjects:
            for s in doc.subjects:
                if isinstance(s, dict):
                    subjects.append(s.get("id", ""))
                elif isinstance(s, str):
                    subjects.append(s)

        tools_called = _extract_tool_calls(
            doc.message_history or [],
        )

        summary: dict = {
            "doc_id": doc.id,
            "subject": subjects[0] if subjects else "unknown",
            "actor": doc.actor_id,
            "tools_called": tools_called,
            "output_snippet": (doc.outcome_summary or "")[:300],
        }

        if doc.aftermath_status == "corrected" and doc.correction_text:
            summary["correction"] = doc.correction_text[:300]
        if doc.parent_doc_id:
            summary["is_correction"] = True

        summaries.append(summary)

    return summaries


def _extract_tool_calls(
    message_history: list[dict],
) -> list[dict]:
    """Extract tool call info with result quality indicators."""
    calls = []
    pending_call: dict | None = None

    for msg in message_history:
        role = msg.get("role", "")
        if role == "tool_call":
            pending_call = {
                "tool": msg.get("name", "unknown"),
                "args": msg.get("arguments", {}),
            }
        elif role == "tool_result" and pending_call:
            content = msg.get("content", "")
            pending_call["result_len"] = len(content)
            pending_call["result_snippet"] = content[:100]

            # Extract result count for web_search JSON
            try:
                parsed = json.loads(content)
                if "results_count" in parsed:
                    pending_call["results_count"] = parsed["results_count"]
            except (json.JSONDecodeError, TypeError):
                pass

            calls.append(pending_call)
            pending_call = None

    return calls


# -- LLM Analyzer ---------------------------------------------


async def analyze_traces(
    trace_summaries: list[dict],
    skill_definition: str = "",
) -> str:
    """LLM reads traces and produces directives + insights.

    Returns injection text with two sections:
    1. Mandatory tool-calling rules (from RULE: lines)
    2. Learned knowledge as few-shot examples (from INSIGHT: lines)
    """
    if not trace_summaries:
        return ""

    from anthropic import AsyncAnthropicVertex

    project = os.environ.get("GCP_PROJECT_ID", "")
    location = os.environ.get(
        "GCP_LOCATION_ANTHROPIC",
        "global",
    )

    client = AsyncAnthropicVertex(
        region=location,
        project_id=project,
    )

    # Strip to tool calls only
    tool_only = []
    for t in trace_summaries:
        if t.get("is_correction"):
            continue
        calls = t.get("tools_called", [])
        if calls:
            tool_only.append(
                {
                    "subject": t.get("subject", "unknown"),
                    "tools_called": calls,
                }
            )

    if not tool_only:
        return ""

    traces_text = json.dumps(tool_only, indent=2)

    system_prompt = (
        "You look at tool call results and produce rules.\n\n"
        "For each tool call you see: tool name, parameters, "
        "result_len, result_snippet, and optionally "
        "results_count.\n\n"
        "Compare ALL tool calls across ALL subjects. Find:\n"
        "- Calls that FAIL (errors, empty) vs calls that "
        "SUCCEED (real data)\n"
        "- Params that return MORE results vs FEWER\n\n"
        "If ALL calls succeed with similar result counts "
        "and no errors, output NO_ISSUES.\n"
        "Do NOT describe normal call order as rules.\n\n"
        "OUTPUT — ONLY these, nothing else:\n"
        "RULE: DO NOT call <tool> with <params> — <reason>. "
        "Use <alternative> instead.\n"
        "RULE: START with <tool>(<params>) — it returns "
        "<what>.\n"
        "NO_ISSUES\n\n"
        "List ALL failing params in one RULE. "
        "Keep exact tool parameter values (source, type). "
        "OMIT specific entity names/domains from query "
        "strings — describe the query type instead."
    )

    user_msg = f"### Execution Traces ###\n{traces_text}"
    if skill_definition:
        user_msg += f"\n\n### Skill Definition ###\n{skill_definition[:500]}"

    try:
        response = await client.messages.create(
            model="claude-sonnet-4@20250514",
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_msg},
            ],
        )

        from lucid_dc.shared.usage import record_llm

        usage = getattr(response, "usage", None)
        if usage is not None:
            record_llm(
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )

        raw = response.content[0].text.strip()
        logger.warning("Analyzer raw:\n%s", raw)

        # Parse RULE: lines first — if any exist, use them
        # even if NO_ISSUES also appears
        rules = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("RULE:"):
                rules.append(line[5:].strip())

        if not rules:
            return ""

        lines = [
            "⚠️ MANDATORY TOOL-CALLING RULES ⚠️",
            "",
        ]
        for r in rules:
            lines.append(r)
        lines.append("")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Analyzer failed: %s", exc)
        return ""


# -- Store and return ------------------------------------------


async def run_consolidation(
    session: AsyncSession,
    skill_name: str,
    skill_definition: str = "",
) -> ConsolidationResult:
    """Run the full consolidation: gather traces → LLM → store."""
    result = ConsolidationResult()

    trace_summaries = await gather_trace_summaries(
        session,
        skill_name,
    )

    if not trace_summaries:
        return result

    injection = await analyze_traces(
        trace_summaries,
        skill_definition,
    )

    if not injection:
        return result

    # Collect subjects from traces
    all_subjects: set[str] = set()
    for t in trace_summaries:
        s = t.get("subject", "")
        if s and s != "unknown":
            all_subjects.add(s)

    # Store as pattern doc
    pattern_id = str(uuid.uuid4())
    subjects_list = [{"id": s, "name": s} for s in sorted(all_subjects)] if all_subjects else []

    doc = Document(
        id=pattern_id,
        skill_name=skill_name,
        skill_version="analyzer",
        actor_id="system-analyzer",
        actor_type="system",
        subjects=subjects_list,
        conversation_id=None,
        parent_doc_id=None,
        timestamp=datetime.now(UTC),
        outcome_summary="[analyzer] directives + insights",
        aftermath_status="corrected",
        correction_text=injection,
        correction_needs_context=False,
        message_history=[],
    )
    session.add(doc)

    result.new_correction_ids.append(pattern_id)
    result.injection_text = injection
    result.llm_corrections_generated = 1

    logger.debug(
        "Stored analyzer doc %s: %s",
        pattern_id[:8],
        injection[:100],
    )

    return result
