"""Positional action_label assignment for reasoning nodes."""

from __future__ import annotations

from lucid_dc.store.capture.splitter import (
    RawNode,
)


def assign_reasoning_labels(
    nodes: list[RawNode],
) -> None:
    """Assign action_labels to reasoning nodes by position.

    Labels are fully deterministic — no text analysis, no LLM.
    Examples:
      before_query_bigquery
      query_bigquery_to_web_search
      after_web_search
      reasoning_0 (no tool calls at all)
    """
    for i, node in enumerate(nodes):
        if node.node_type == "tool":
            continue

        prev_tool = _find_prev_tool(nodes, i)
        next_tool = _find_next_tool(nodes, i)

        if prev_tool and next_tool:
            node.action_label = f"{prev_tool.tool_name}_to_{next_tool.tool_name}"
        elif next_tool:
            node.action_label = f"before_{next_tool.tool_name}"
        elif prev_tool:
            node.action_label = f"after_{prev_tool.tool_name}"
        else:
            node.action_label = f"reasoning_{i}"


def _find_prev_tool(nodes: list[RawNode], idx: int) -> RawNode | None:
    for j in range(idx - 1, -1, -1):
        if nodes[j].node_type == "tool":
            return nodes[j]
    return None


def _find_next_tool(nodes: list[RawNode], idx: int) -> RawNode | None:
    for j in range(idx + 1, len(nodes)):
        if nodes[j].node_type == "tool":
            return nodes[j]
    return None
