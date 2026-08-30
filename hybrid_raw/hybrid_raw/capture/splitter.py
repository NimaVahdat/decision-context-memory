"""Split message history into reasoning + tool nodes.

Groups messages by tool call:
buffer messages, flush on tool_result. Each flush
becomes a tool node. Remaining buffer (final
assistant output) becomes a reasoning node.

This aligns capture with injection — per-tool-call
injection appends context after each tool_result,
so each tool node maps 1:1 to an injection point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RawNode:
    """Intermediate representation before DB storage."""

    node_type: str  # "tool" or "reasoning"
    content: str
    tool_name: str | None = None
    action_label: str | None = None
    evidence_content: str | None = None


def split_message_history(
    messages: list[dict],
) -> list[RawNode]:
    """Split messages into tool-group nodes.

    Buffers messages and flushes on each tool_result,
    matching the injection boundary. Each flush
    produces a tool node containing any preceding
    reasoning + the tool_call + tool_result.

    Remaining buffer after all tool_results (typically
    the final assistant output) becomes a reasoning
    node.
    """
    nodes: list[RawNode] = []
    buffer: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            continue

        buffer.append(msg)

        # Flush on tool_result — one tool node
        if role == "tool_result":
            nodes.append(_flush_tool_node(buffer))
            buffer = []

    # Remaining buffer → reasoning node (assistant only)
    if buffer:
        assistant_msgs = [m for m in buffer if m.get("role") == "assistant"]
        if assistant_msgs:
            content = _extract_content(assistant_msgs)
            if content.strip():
                nodes.append(
                    RawNode(
                        node_type="reasoning",
                        content=content,
                        tool_name=None,
                        action_label=None,
                    )
                )

    return nodes


def _flush_tool_node(buffer: list[dict]) -> RawNode:
    """Create a tool node from a buffered group.

    Extracts tool_name from the tool_call message,
    evidence from the tool_result, and builds
    content from all messages in the group.
    """
    tool_name = ""
    evidence = ""
    content_parts: list[str] = []

    for msg in buffer:
        role = msg.get("role", "")

        if role == "tool_call":
            tool_name = msg.get("name", "")
            args = msg.get("arguments", {})
            content_parts.append(f"[{tool_name}({args})]")
        elif role == "tool_result":
            evidence = msg.get("content", "")
        elif role in ("user", "assistant"):
            text = msg.get("content", "")
            if text.strip():
                content_parts.append(text)

    content = "\n".join(content_parts)

    return RawNode(
        node_type="tool",
        content=content,
        tool_name=tool_name,
        action_label=tool_name,
        evidence_content=evidence or None,
    )


def _extract_content(
    messages: list[dict],
) -> str:
    """Extract text content from messages."""
    parts: list[str] = []
    for msg in messages:
        text = msg.get("content", "")
        if text.strip():
            parts.append(text)
    return "\n".join(parts)
