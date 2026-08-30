"""The forge pipeline: chains → LLM compilation → stored rules.

Phase 1 (deterministic): gather docs, walk chains, extract tool data
Phase 2 (parallel LLM): haiku tool analysis + strong model compilation
Phase 3 (store): parse rules, embed, store as DecisionNodes + edges
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from forge_dc.capture.models import (
    CompiledRule,
    ForgeResult,
    RuleScope,
    RuleSource,
)
from forge_dc.store.models import (
    DecisionNode,
    Document,
    Edge,
)

logger = logging.getLogger(__name__)


# ── Phase 1: Deterministic ──────────────────────────────────


async def gather_all_docs(
    session: AsyncSession,
    skill_name: str,
    conversation_id: str | None = None,
) -> list[dict]:
    """Fetch Documents + their DecisionNodes for this skill.

    If conversation_id is provided, fetches only docs in that
    conversation — scopes compilation to one correction chain.
    """
    stmt = (
        select(Document)
        .where(Document.skill_name == skill_name)
        .where(Document.actor_id != "forge-compiler")
        .order_by(Document.timestamp.asc())
    )
    if conversation_id:
        stmt = stmt.where(Document.conversation_id == conversation_id)
    result = await session.execute(stmt)
    docs = list(result.scalars().all())

    out = []
    for doc in docs:
        subjects = []
        if doc.subjects:
            for s in doc.subjects:
                if isinstance(s, dict):
                    subjects.append(s.get("id", ""))
                elif isinstance(s, str):
                    subjects.append(s)

        # Load decision nodes for this doc
        node_stmt = (
            select(DecisionNode)
            .where(DecisionNode.document_id == doc.id)
            .where(DecisionNode.node_type != "forge_rule")
            .order_by(DecisionNode.decision_index)
        )
        node_result = await session.execute(node_stmt)
        nodes = list(node_result.scalars().all())

        out.append(
            {
                "doc_id": doc.id,
                "actor_id": doc.actor_id,
                "subjects": subjects,
                "parent_doc_id": doc.parent_doc_id,
                "conversation_id": doc.conversation_id,
                "aftermath_status": doc.aftermath_status,
                "correction_text": doc.correction_text,
                "correction_needs_context": doc.correction_needs_context,
                "outcome_summary": doc.outcome_summary or "",
                "message_history": doc.message_history or [],
                "timestamp": (doc.timestamp.isoformat() if doc.timestamp else ""),
                "nodes": [
                    {
                        "id": n.id,
                        "idx": n.decision_index,
                        "type": n.node_type,
                        "content": n.content[:200] if n.content else "",
                        "tool_name": n.tool_name,
                        "action_label": n.action_label,
                    }
                    for n in nodes
                ],
            }
        )

    return out


def build_chains(docs: list[dict]) -> list[list[dict]]:
    """Build correction chains from conversation grouping.

    Groups docs by conversation_id, orders by timestamp.
    Splits chains at "accepted" docs — an accepted doc means
    the chain ended and a new request follows.

    This matches the aftermath classifier's behavior:
    - "corrected" → chain continues (next doc is a correction)
    - "accepted" → chain ends (next doc is a new request)
    - "pending" → no follow-up yet
    """
    by_conv: dict[str, list[dict]] = {}
    no_conv: list[dict] = []

    for d in docs:
        cid = d.get("conversation_id")
        if cid:
            by_conv.setdefault(cid, []).append(d)
        else:
            no_conv.append(d)

    chains: list[list[dict]] = []

    for cid, conv_docs in by_conv.items():
        conv_docs.sort(
            key=lambda d: d.get("timestamp", ""),
        )

        # Split at "accepted" docs — they mark chain boundaries
        current_chain: list[dict] = []
        for d in conv_docs:
            current_chain.append(d)
            if d["aftermath_status"] == "accepted":
                # Chain ends here. Next doc starts new chain.
                chains.append(current_chain)
                current_chain = []

        if current_chain:
            chains.append(current_chain)

    for d in no_conv:
        chains.append([d])

    return chains


def extract_tool_data(docs: list[dict]) -> list[dict]:
    """Extract tool call summaries for haiku analysis.

    Strips to tool-call data only: only tool call data,
    no corrections or outputs.
    """
    from forge_dc.analysis.consolidate import (
        _extract_tool_calls,
    )

    tool_only = []
    for d in docs:
        if d["aftermath_status"] == "corrected" and d.get(
            "parent_doc_id",
        ):
            continue
        calls = _extract_tool_calls(d["message_history"])
        if calls:
            subject = d["subjects"][0] if d["subjects"] else "unknown"
            tool_only.append(
                {
                    "subject": subject,
                    "tools_called": calls,
                }
            )

    return tool_only


def format_chains_for_llm(
    chains: list[list[dict]],
) -> tuple[str, dict[int, list[str]], dict[int, str], list[int], dict[str, str]]:
    """Format correction chains for the strong model prompt.

    Returns:
        prompt_text: the formatted chains for the user message
        chain_doc_map: {chain_index: [doc_ids]} for linking rules
        chain_actor_map: {chain_index: actor_id} for scope detection
        corr_chain_seq: [chain_idx, ...] one per correction in order
        node_id_map: {"N0": actual_node_uuid, ...} for TRIGGER edges
    """
    chain_doc_map: dict[int, list[str]] = {}
    chain_actor_map: dict[int, str] = {}
    corr_chain_seq: list[int] = []
    node_id_map: dict[str, str] = {}  # "N0" → node UUID
    sections: list[str] = []
    chain_idx = 0

    for chain in chains:
        corrections = [
            d for d in chain if d["aftermath_status"] == "corrected" and d["correction_text"]
        ]
        if not corrections:
            continue

        doc_ids = [d["doc_id"] for d in chain]
        chain_doc_map[chain_idx] = doc_ids

        # Gather metadata from chain
        actors = {d["actor_id"] for d in chain if d["actor_id"]}
        # Track primary actor (the one who made corrections)
        correction_actors = {d["actor_id"] for d in corrections if d["actor_id"]}
        primary_actor = sorted(correction_actors)[0] if correction_actors else "unknown"
        chain_actor_map[chain_idx] = primary_actor
        subjects: set[str] = set()
        for d in chain:
            subjects.update(d["subjects"])

        actor_str = ", ".join(sorted(actors)) if actors else "unknown"
        subj_str = "[" + ", ".join(sorted(subjects)) + "]" if subjects else "[]"

        # Build node listing for all docs in chain
        node_lines = []
        node_counter = len(node_id_map)
        for d in chain:
            for n in d.get("nodes", []):
                label = f"N{node_counter}"
                node_id_map[label] = n["id"]
                ntype = n["type"]
                content = n["content"][:150]
                if n.get("tool_name"):
                    node_lines.append(f'  {label}: {ntype} {n["tool_name"]} → "{content}"')
                else:
                    node_lines.append(f'  {label}: {ntype} → "{content}"')
                node_counter += 1

        if len(corrections) == 1 and len(chain) <= 2:
            # Standalone correction
            c = corrections[0]
            header = f"=== STANDALONE (chain {chain_idx}) ==="
            header += f"\nactor: {actor_str} | subjects: {subj_str}"

            text = c["correction_text"]
            if c.get("correction_needs_context"):
                # Find parent outcome for context
                parent_id = c.get("parent_doc_id")
                parent_outcome = ""
                for d in chain:
                    if d["doc_id"] == parent_id:
                        parent_outcome = d["outcome_summary"][:300]
                        break
                if parent_outcome:
                    header += (
                        f'\nCorrection: "{text}"\nContext from original output: {parent_outcome}'
                    )
                else:
                    header += f'\n"{text}"'
            else:
                header += f'\n"{text}"'

            if node_lines:
                header += "\nTrace nodes:\n"
                header += "\n".join(node_lines)
            corr_chain_seq.append(chain_idx)
            sections.append(header)
        else:
            # Multi-correction chain
            header = f"=== CHAIN {chain_idx} ==="
            header += f"\nactor: {actor_str} | subjects: {subj_str}"

            if node_lines:
                header += "\nTrace nodes:\n"
                header += "\n".join(node_lines)

            header += "\nCorrections (in order):"
            for i, c in enumerate(corrections, 1):
                ctext = c["correction_text"]
                line = f'{i}. "{ctext}"'
                if c.get("correction_needs_context"):
                    parent_id = c.get("parent_doc_id")
                    for d in chain:
                        if d["doc_id"] == parent_id:
                            line += f"\n   Context: {d['outcome_summary'][:200]}"
                            break
                header += f"\n{line}"
                corr_chain_seq.append(chain_idx)

            sections.append(header)

        chain_idx += 1

    prompt_text = "Compile these correction chains into RULE directives.\n\n" + "\n\n".join(
        sections
    )

    return (
        prompt_text,
        chain_doc_map,
        chain_actor_map,
        corr_chain_seq,
        node_id_map,
    )


# ── Phase 2: LLM Calls ──────────────────────────────────────


FORGE_SYSTEM_PROMPT = """\
You are a rule compiler for an AI agent system.

PURPOSE: You produce rules that will be injected into FUTURE \
agent runs — different queries, different subjects, different \
contexts. The agent receiving these rules has NEVER seen the \
original conversation. It has NO knowledge of the correction \
chain, the previous outputs, or the specific entities involved. \
Every rule you produce must make sense to an agent seeing it \
for the first time with zero context. If a correction says \
"X instead of Y" — the agent doesn't know Y existed. Write \
only "X".

The input shows TRACE NODES (Nx) from a past agent run, \
followed by CORRECTIONS the human made.

CRITICAL: produce one or more RULE lines per correction. \
SPLIT into multiple rules only when the parts are independently \
enforceable at different points in the agent's execution — \
each part gets its own RULE and its own TRIGGER. \
KEEP as ONE rule when the parts are co-dependent: \
causally linked, conditional, or a substitution where \
splitting destroys meaning (e.g. "use X — it equals Y/4" \
must stay together). Never silently omit a correction.

FOR EACH CORRECTION produce:
- RULE: if the correction is already a clear imperative \
directive, PRESERVE it exactly. Only rewrite if the \
correction is conversational, references specific entities, \
or needs decontextualization.
- TYPE: FORMAT or CONTENT.
  FORMAT = applies to ALL outputs regardless of topic \
(units, structure, sections, labels, styling).
  CONTENT = depends on specific data or conditions.
- TRIGGER: Nx — the trace node whose data makes this rule \
relevant. Think: "What is the LAST thing the agent saw \
BEFORE making the mistake?" For tool failures: the trigger \
is NOT the failure itself — it is the data BEFORE the \
failing call. What should have warned it?
- DROP: if a LATER correction supersedes it.
- SKIP: if too context-specific to generalize.

GROUPING:
- Wrap RELATED corrections in GROUP: ... END_GROUP
- Related = about the same output area.
- UNRELATED corrections go in SEPARATE groups.

EXAMPLE:
Trace nodes:
  N0: tool_call get_weather → "25°C, wind 40km/h, rain 20%"
  N1: output → "Temperature is 77°F. Expect clear skies."

Corrections:
1. "Show temperatures in Celsius, not Fahrenheit"
2. "Add wind speed to the summary"
3. "Keep it to a short paragraph"
4. "Actually use a table instead"

Output:
GROUP:
RULE: USE Celsius for temperatures, not Fahrenheit.
TYPE: CONTENT
TRIGGER: N0
RULE: INCLUDE wind speed in the summary.
TYPE: CONTENT
TRIGGER: N0
END_GROUP
GROUP:
DROP: superseded by #4
RULE: FORMAT the output as a table.
TYPE: FORMAT
TRIGGER: N1
END_GROUP

OUTPUT: RULE/TYPE/TRIGGER/DROP/SKIP in GROUP/END_GROUP blocks.\
"""


# Reasoning-variant prompt (enabled when emit_reasoning=True).
# Adds WHY: <one-line justification> per rule.
FORGE_SYSTEM_PROMPT_WHY = """\
You are a rule compiler for an AI agent system.

PURPOSE: You produce rules that will be injected into FUTURE \
agent runs — different queries, different subjects, different \
contexts. The agent receiving these rules has NEVER seen the \
original conversation. Every rule must make sense to an agent \
seeing it for the first time with zero context. If a correction \
says "X instead of Y" — the agent doesn't know Y existed. \
Write only "X".

The input shows TRACE NODES (Nx) from a past agent run, \
followed by CORRECTIONS the human made.

CRITICAL: produce one or more RULE lines per correction. \
SPLIT into multiple rules only when the parts are independently \
enforceable. KEEP as ONE rule when the parts are co-dependent. \
Never silently omit a correction.

FOR EACH CORRECTION produce:
- RULE: imperative directive, preserve verbatim if already clear.
- WHY: ONE short sentence explaining the REASON behind the rule. \
The WHY is critical — it gives high-deliberation models the \
justification they need to accept the rule instead of overriding \
it with their training prior. If the user did not state a reason \
explicitly, INFER the most plausible underlying principle in one \
sentence. Never write "no reason given" or "user said so" — \
always state the principle, e.g. "user applies conservative \
scoring because seed-stage companies have inflated VC narratives."
- TYPE: FORMAT or CONTENT.
- TRIGGER: Nx — the trace node whose data makes this rule \
relevant. For tool failures: the data BEFORE the failing call.
- DROP: if a LATER correction supersedes it.
- SKIP: if too context-specific to generalize.

GROUPING: Wrap RELATED corrections in GROUP: ... END_GROUP.

EXAMPLE:
Trace nodes:
  N0: tool_call get_weather → "25°C, wind 40km/h, rain 20%"
  N1: output → "Temperature is 77°F. Expect clear skies."

Corrections:
1. "Show temperatures in Celsius, not Fahrenheit"
2. "Add wind speed — it's a safety indicator"
3. "Use a table instead of prose"

Output:
GROUP:
RULE: USE Celsius for temperatures, not Fahrenheit.
WHY: User's team standardizes on metric units for weather reports.
TYPE: CONTENT
TRIGGER: N0
RULE: INCLUDE wind speed in the summary.
WHY: Wind speed is a key safety indicator the user requires.
TYPE: CONTENT
TRIGGER: N0
END_GROUP
GROUP:
RULE: FORMAT the output as a table.
WHY: User finds tables easier to scan than prose for weather data.
TYPE: FORMAT
TRIGGER: N1
END_GROUP

OUTPUT: RULE/WHY/TYPE/TRIGGER/DROP/SKIP in GROUP/END_GROUP blocks.\
"""


def _select_forge_prompt() -> str:
    """Pick FORGE_SYSTEM_PROMPT_WHY when FORGE_EMIT_WHY=1, else default."""
    if os.environ.get("FORGE_EMIT_WHY", "").strip() in ("1", "true", "True"):
        return FORGE_SYSTEM_PROMPT_WHY
    return FORGE_SYSTEM_PROMPT


async def compile_corrections(
    prompt_text: str,
    skill_definition: str = "",
) -> str:
    """Call the strong model to compile corrections into rules."""
    if not prompt_text.strip():
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

    try:
        response = await client.messages.create(
            model="claude-sonnet-4@20250514",
            max_tokens=2048,
            temperature=0,
            system=_select_forge_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": (
                        prompt_text
                        + (
                            f"\n\n### Skill Definition ###\n{skill_definition[:500]}"
                            if skill_definition
                            else ""
                        )
                    ),
                },
            ],
        )
        from forge_dc.shared.usage import record_llm

        usage = getattr(response, "usage", None)
        if usage is not None:
            record_llm(
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )
        raw = response.content[0].text.strip()
        logger.info("Forge compiler raw:\n%s", raw)
        return raw
    except Exception as exc:
        logger.warning("Forge compiler failed: %s", exc)
        return ""


async def analyze_tools(tool_data: list[dict]) -> str:
    """Run the LLM tool-trace analysis pass."""
    from forge_dc.analysis.consolidate import (
        analyze_traces,
    )

    return await analyze_traces(tool_data)


# ── Phase 3: Parse and Store ─────────────────────────────────


_RULE_RE = re.compile(
    r"^RULE\s*"
    r"(?:\[([^\]]*)\])?"
    r"\s*:?\s*(.+)$",
)


def parse_rules(
    raw_tool: str,
    raw_correction: str,
    chain_doc_map: dict[int, list[str]],
    chain_actor_map: dict[int, str],
    corr_chain_seq: list[int],
    node_id_map: dict[str, str],
    skill_name: str,
) -> list[CompiledRule]:
    """Parse RULE lines from both LLM outputs.

    All rules are GLOBAL. Actor preference is resolved at
    injection time via ownership tags, not scope filtering.
    """
    rules: list[CompiledRule] = []

    # Parse tool rules (always GLOBAL)
    # analyze_traces strips the RULE: prefix, so handle
    # both "RULE: text" and plain directive lines.
    if raw_tool:
        for line in raw_tool.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("\u26a0") or line.startswith(
                "NO_ISSUES",
            ):
                continue

            m = _RULE_RE.match(line)
            if m:
                text = m.group(2).strip()
            elif line.startswith("RULE:"):
                text = line[5:].strip()
            else:
                text = line

            if text:
                rules.append(
                    CompiledRule(
                        id=str(uuid.uuid4()),
                        text=text,
                        scope=RuleScope.GLOBAL,
                        source=RuleSource.TOOL,
                        skill_name=skill_name,
                    )
                )

    # Parse correction rules with GROUP markers.
    # Each GROUP section maps to a chain in prompt order.
    if raw_correction:
        chain_keys = sorted(chain_doc_map.keys())
        group_idx = -1
        current_group = str(uuid.uuid4().hex[:8])

        for line in raw_correction.split("\n"):
            line = line.strip()
            if not line:
                continue

            upper = line.upper()

            # GROUP markers — each GROUP = next chain section
            if upper.startswith("GROUP"):
                group_idx += 1
                current_group = str(uuid.uuid4().hex[:8])
                continue
            if upper.startswith("END_GROUP"):
                continue

            # TYPE: FORMAT or CONTENT — assign to LAST rule
            if upper.startswith("TYPE"):
                type_text = line.split(":", 1)[-1].strip().lower()
                if rules and type_text in ("format", "content"):
                    rules[-1].rule_type = type_text
                continue

            # TRIGGER: Nx — assign to the LAST appended rule
            if upper.startswith("TRIGGER"):
                trigger_text = line.split(":", 1)[-1].strip()
                import re as _re

                nm = _re.search(r"N(\d+)", trigger_text)
                if nm and rules:
                    label = f"N{nm.group(1)}"
                    node_id = node_id_map.get(label)
                    if node_id:
                        rules[-1].trigger_node_id = node_id
                continue

            # WHY: justification — assign to the LAST appended rule.
            # Only present when FORGE_EMIT_WHY=1 selected the WHY prompt.
            if upper.startswith("WHY"):
                why_text = line.split(":", 1)[-1].strip()
                if why_text and rules:
                    rules[-1].reasoning = why_text
                continue

            # DROP/SKIP: don't create rule
            if upper.startswith("DROP") or upper.startswith(
                "SKIP",
            ):
                continue

            if not upper.startswith("RULE"):
                continue

            m = _RULE_RE.match(line)
            if m:
                text = m.group(2).strip()
            elif line.startswith("RULE:"):
                text = line[5:].strip()
            else:
                continue

            if not text:
                continue

            # Map to chain via GROUP index
            chain_i = chain_keys[group_idx] if 0 <= group_idx < len(chain_keys) else 0
            actor_id = chain_actor_map.get(chain_i)
            source_doc_ids = list(
                chain_doc_map.get(chain_i, []),
            )

            rules.append(
                CompiledRule(
                    id=str(uuid.uuid4()),
                    text=text,
                    scope=RuleScope.GLOBAL,
                    source=RuleSource.CORRECTION,
                    skill_name=skill_name,
                    topic_group=current_group,
                    actor_id=actor_id,
                    source_doc_ids=source_doc_ids,
                )
            )

    return rules


def assign_tool_triggers(
    rules: list[CompiledRule],
    docs: list[dict],
) -> None:
    """Assign trigger nodes to tool rules deterministically.

    For each tool rule: find a matching tool call node in
    traces. Link to the node BEFORE it (the precondition).

    Matching strategy:
    1. Tool name match (strongest — rule mentions the tool)
    2. Keyword overlap on content (weaker — rule keywords in
       node content)
    """
    all_tool_nodes: list[tuple[dict, int, list[dict]]] = []
    for doc in docs:
        nodes = doc.get("nodes", [])
        for i, node in enumerate(nodes):
            if node.get("type") == "tool":
                all_tool_nodes.append((node, i, nodes))

    for rule in rules:
        if rule.source != RuleSource.TOOL:
            continue
        if rule.trigger_node_id:
            continue

        rule_lower = rule.text.lower()

        best_match = None
        best_score = 0

        for node, idx, siblings in all_tool_nodes:
            tool_name = node.get("tool_name", "") or ""
            action = node.get("action_label", "") or ""
            content = node.get("content", "") or ""

            score = 0

            # Tool name match (strong signal)
            if tool_name and tool_name.lower() in rule_lower:
                score += 10

            # Action label match
            if action and action.lower() in rule_lower:
                score += 5

            # Keyword overlap (weaker)
            rule_words = set(rule_lower.split())
            node_words = set(f"{tool_name} {action} {content}".lower().split())
            score += len(rule_words & node_words)

            if score > best_score:
                best_score = score
                if idx > 0:
                    best_match = siblings[idx - 1]["id"]
                else:
                    best_match = node["id"]

        if best_match and best_score >= 2:
            rule.trigger_node_id = best_match


async def delete_forge_doc_for_conversation(
    session: AsyncSession,
    skill_name: str,
    conversation_id: str,
) -> None:
    """Delete the existing forge rulebook for this conversation.

    Deletes in dependency order: FORGED_FROM edges → DecisionNodes → Document.
    Does not touch forge docs from other conversations.
    """
    from sqlalchemy import delete as sa_delete

    # Find existing forge doc for this conversation
    stmt = (
        select(Document.id)
        .where(Document.skill_name == skill_name)
        .where(Document.actor_id == "forge-compiler")
        .where(Document.conversation_id == conversation_id)
    )
    result = await session.execute(stmt)
    forge_doc_ids = [r[0] for r in result.fetchall()]

    if not forge_doc_ids:
        return

    # Collect all rule node IDs for these docs
    node_stmt = select(DecisionNode.id).where(DecisionNode.document_id.in_(forge_doc_ids))
    node_result = await session.execute(node_stmt)
    node_ids = [r[0] for r in node_result.fetchall()]

    # Delete FORGED_FROM edges originating from these rule nodes
    if node_ids:
        await session.execute(sa_delete(Edge).where(Edge.from_id.in_(node_ids)))

    # Delete rule nodes
    await session.execute(
        sa_delete(DecisionNode).where(DecisionNode.document_id.in_(forge_doc_ids))
    )

    # Delete forge documents
    await session.execute(sa_delete(Document).where(Document.id.in_(forge_doc_ids)))

    logger.info(
        "Deleted %d old forge doc(s) for conversation %s",
        len(forge_doc_ids),
        conversation_id,
    )


async def store_rules(
    session: AsyncSession,
    rules: list[CompiledRule],
    skill_name: str,
    embedder,
    conversation_id: str | None = None,
) -> str:
    """Store compiled rules as DecisionNodes + FORGED_FROM edges.

    Returns the forge document ID.
    """
    forge_doc_id = str(uuid.uuid4())

    # Create parent document
    doc = Document(
        id=forge_doc_id,
        skill_name=skill_name,
        skill_version="forge-rulebook",
        actor_id="forge-compiler",
        actor_type="system",
        subjects=[],
        conversation_id=conversation_id,
        parent_doc_id=None,
        timestamp=datetime.now(UTC),
        outcome_summary="[forge] compiled rulebook",
        aftermath_status="corrected",
        correction_text="",
        correction_needs_context=False,
        message_history=[
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "rules_count": len(rules),
                        "rules": [r.to_dict() for r in rules],
                    }
                ),
            }
        ],
    )
    session.add(doc)
    await session.flush()  # FK reference must exist before nodes

    # Embed all rule texts in batch
    rule_texts = [r.text for r in rules]
    embeddings: list[list[float]] = []
    if embedder and rule_texts:
        try:
            embeddings = await embedder.embed_batch(rule_texts)
        except Exception as exc:
            logger.warning("Rule embedding failed: %s", exc)
            embeddings = [[] for _ in rule_texts]
    else:
        embeddings = [[] for _ in rule_texts]

    # Create DecisionNodes for each rule
    for i, rule in enumerate(rules):
        node_id = rule.id
        node = DecisionNode(
            id=node_id,
            document_id=forge_doc_id,
            node_type="forge_rule",
            content=rule.text,
            embedding=embeddings[i] if i < len(embeddings) else [],
            action_label="forge_rule",
            tool_name=None,
            subjects=rule.subjects,
            evidence_summary=None,
            outcome_short=json.dumps(
                {
                    "scope": rule.scope.value,
                    "actor_id": rule.actor_id,
                    "supported_by": rule.supported_by,
                    "conflict_group": rule.conflict_group,
                    "topic_group": rule.topic_group,
                    "trigger_node_id": rule.trigger_node_id,
                    "rule_type": rule.rule_type,
                }
            ),
            decision_index=i,
            # Persist the reasoning WHY into the reason column.
            # For default forge_dc this stays None (reasoning is "").
            reason=(rule.reasoning or None),
        )
        session.add(node)

        # FORGED_FROM edge to trigger node (precise)
        # Falls back to nodes of source documents when
        # the compiler didn't output a TRIGGER line.
        if rule.trigger_node_id:
            edge = Edge(
                id=str(uuid.uuid4()),
                edge_type="FORGED_FROM",
                from_id=node_id,
                to_id=rule.trigger_node_id,
            )
            session.add(edge)
        elif rule.source_doc_ids:
            # Resolve document IDs to their decision nodes
            # so edge following (which matches node IDs)
            # can find these rules.
            for source_doc_id in rule.source_doc_ids:
                node_stmt = select(DecisionNode.id).where(
                    DecisionNode.document_id == source_doc_id,
                )
                node_result = await session.execute(node_stmt)
                doc_node_ids = [r[0] for r in node_result.fetchall()]
                for doc_node_id in doc_node_ids:
                    edge = Edge(
                        id=str(uuid.uuid4()),
                        edge_type="FORGED_FROM",
                        from_id=node_id,
                        to_id=doc_node_id,
                    )
                    session.add(edge)

    return forge_doc_id


# ── Main Pipeline ────────────────────────────────────────────


async def run_forge_pipeline(
    session: AsyncSession,
    skill_name: str,
    skill_definition: str = "",
    embedder=None,
    conversation_id: str | None = None,
) -> ForgeResult:
    """Run the full forge pipeline.

    Phase 1: Gather docs for this conversation, walk chains, extract tool data
    Phase 2: Parallel LLM (haiku tools + strong model corrections)
    Phase 3: Delete old forge doc for this conversation, store new rules
    """
    result = ForgeResult()

    # Phase 1 — gather all raw traces for this skill (not scoped to conversation)
    docs = await gather_all_docs(session, skill_name)
    if not docs:
        return result

    chains = build_chains(docs)
    tool_data = extract_tool_data(docs)

    # Group chains by actor — compile each actor separately
    # so the compiler can't merge different actors' preferences
    # into one rule. Maintenance detects conflicts across actors.
    chains_by_actor: dict[str, list[list[dict]]] = defaultdict(list)
    for chain in chains:
        correction_actors = {
            d["actor_id"]
            for d in chain
            if d["aftermath_status"] == "corrected" and d["correction_text"] and d["actor_id"]
        }
        actor = sorted(correction_actors)[0] if correction_actors else "unknown"
        chains_by_actor[actor].append(chain)

    # Format per-actor
    actor_compilations: dict[str, dict] = {}
    for actor, actor_chains in chains_by_actor.items():
        fmt = format_chains_for_llm(actor_chains)
        prompt_text, cdm, cam, ccs, nim = fmt
        if cdm:  # has corrections
            actor_compilations[actor] = {
                "prompt": prompt_text,
                "chain_doc_map": cdm,
                "chain_actor_map": cam,
                "corr_chain_seq": ccs,
                "node_id_map": nim,
            }

    has_corrections = bool(actor_compilations)
    has_tools = bool(tool_data)

    logger.info(
        "Forge Phase 1: %d docs, %d chains, %d actors, %d tool traces",
        len(docs),
        len(chains),
        len(actor_compilations),
        len(tool_data),
    )

    # Phase 2: parallel LLM calls — one per actor + tools
    llm_tasks: dict[str, Any] = {}
    if has_corrections:
        for actor, comp in actor_compilations.items():
            llm_tasks[f"corr_{actor}"] = compile_corrections(
                comp["prompt"],
                skill_definition,
            )
    if has_tools:
        llm_tasks["tools"] = analyze_tools(tool_data)

    if not llm_tasks:
        return result

    raw_results = await asyncio.gather(*llm_tasks.values())
    result_map = dict(zip(llm_tasks.keys(), raw_results))

    # Phase 3: parse per-actor, then combine
    rules: list[CompiledRule] = []

    # Tool rules
    raw_tool = result_map.get("tools", "")
    if raw_tool:
        rules.extend(
            parse_rules(
                raw_tool,
                "",
                {},
                {},
                [],
                {},
                skill_name,
            )
        )

    # Correction rules — per-actor compilation preserves
    # actor_id correctly (no cross-actor GROUP merging)
    for actor, comp in actor_compilations.items():
        raw = result_map.get(f"corr_{actor}", "")
        if raw:
            rules.extend(
                parse_rules(
                    "",
                    raw,
                    comp["chain_doc_map"],
                    comp["chain_actor_map"],
                    comp["corr_chain_seq"],
                    comp["node_id_map"],
                    skill_name,
                )
            )

    if not rules:
        return result

    # Assign trigger nodes to tool rules (precondition node)
    assign_tool_triggers(rules, docs)

    # Embed rules for maintenance comparison
    if embedder:
        texts = [r.text for r in rules]
        try:
            embs = await embedder.embed_batch(texts)
            for r, emb in zip(rules, embs):
                r._embedding = emb
        except Exception:
            pass

    # Rulebook maintenance: dedup, supersession, conflicts
    from forge_dc.capture.maintenance import (
        maintain_rulebook,
    )

    rules = maintain_rulebook(
        new_rules=rules,
        existing_rules=[],
    )

    # Delete previous forge rulebook for this conversation before
    # storing the new one — prevents stale rule accumulation.
    if conversation_id:
        await delete_forge_doc_for_conversation(
            session,
            skill_name,
            conversation_id,
        )

    forge_doc_id = await store_rules(
        session,
        rules,
        skill_name,
        embedder,
        conversation_id=conversation_id,
    )

    result.rules = rules
    result.forge_doc_id = forge_doc_id
    result.stats = {
        "docs_analyzed": len(docs),
        "chains_found": len(chains),
        "tool_traces": len(tool_data),
        "rules_compiled": len(rules),
        "tool_rules": sum(1 for r in rules if r.source == RuleSource.TOOL),
        "correction_rules": sum(1 for r in rules if r.source == RuleSource.CORRECTION),
    }

    logger.info("Forge complete: %s", result.stats)

    return result
