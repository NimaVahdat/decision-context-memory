"""Deterministic annotation of 5 fields per node."""

from __future__ import annotations

import re

from forge_dc.store.capture.splitter import (
    RawNode,
)


def annotate_nodes(
    nodes: list[RawNode],
    subjects: list[str],
) -> list[dict]:
    """Annotate each node with 5 deterministic fields.

    Returns dicts ready for DecisionNode creation.
    """
    annotated = []
    for idx, node in enumerate(nodes):
        evidence_summary = ""
        if node.node_type == "tool" and node.evidence_content:
            evidence_summary = node.evidence_content[:200]

        outcome_short = _extract_outcome(node.content)

        annotated.append(
            {
                "node_type": node.node_type,
                "content": node.content,
                "tool_name": node.tool_name,
                "action_label": node.action_label,
                "subjects": subjects,
                "evidence_summary": evidence_summary,
                "outcome_short": outcome_short,
                "decision_index": idx,
                "evidence_content": (node.evidence_content),
            }
        )

    return annotated


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
