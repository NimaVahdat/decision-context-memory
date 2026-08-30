"""Pre-formatted text for prompt injection into OpenGeorge.

Formatting lives here (co-located with the data model) so
changes to format don't require OG deploys.
"""

from __future__ import annotations


def format_global_readout(readout: dict) -> str:
    """Format a GlobalReadout dict as prompt-ready text.

    Deduplicates correction patterns across action
    labels — the same correction appearing on multiple
    actions is only shown once.

    Returns an empty string when there is nothing to show.
    """
    skill = readout.get("skill_name", "unknown")
    total = readout.get("total_runs", 0)
    actions = readout.get("actions", [])

    if total == 0 or not actions:
        return ""

    lines = [
        f"## Past Experience ({total} total runs for {skill})",
        "",
    ]

    seen_patterns: set[str] = set()

    for a in actions:
        label = a.get("action_label", "?")
        runs = a.get("runs", 0)
        corrections = a.get("corrections", 0)
        rate = a.get("correction_rate", 0.0)
        settled = a.get("is_settled", False)
        patterns = a.get("correction_patterns", [])

        # Dedup patterns already shown by another action
        new_patterns = [p for p in patterns if p not in seen_patterns]

        status = "settled" if settled else f"{corrections} corrections ({rate:.0%})"
        line = f"- {label}: {runs} runs, {status}"
        lines.append(line)

        for p in new_patterns[:3]:
            lines.append(f'  Common correction: "{p}"')
            seen_patterns.add(p)

    return "\n".join(lines)


def format_precedents(
    result: dict,
    mode: str = "raw",
    actor_id: str | None = None,
) -> str:
    """Format a precedents query result as prompt-ready text.

    Args:
        result: Precedents query result dict.
        mode: Injection format mode.
            "structured" — lean by default, but falls
                back to raw for corrections flagged as
                needing context.
            "raw" — full dump: all fields included.
        actor_id: When set, corrections from this actor
            are tagged "(your correction)" and others
            "(other user)" so the agent can prioritize.

    Returns an empty string when there are no results.
    """
    precedents = result.get("results", [])
    if not precedents:
        return ""

    lines = ["## Relevant Precedents", ""]

    for p in precedents:
        aftermath = p.get("aftermath", "pending")
        label = p.get("action_label", "?")
        correction = p.get("correction")
        corrected = p.get("corrected_outcome")
        content = p.get("content", "")
        chain_len = p.get("chain_length", 0)
        chain_pos = p.get("chain_position", 0)
        is_latest = p.get("is_latest", False)

        doc = p.get("document_metadata", {})
        needs_ctx = doc.get(
            "correction_needs_context",
            False,
        )

        subjects = doc.get("subjects") or result.get("subjects", [])
        subject_parts: list[str] = []
        for s in subjects or []:
            if isinstance(s, str):
                subject_parts.append(s)
            elif isinstance(s, dict):
                subject_parts.append(
                    s.get("name") or s.get("id", ""),
                )
        subject_str = ", ".join(subject_parts)

        # Chain-grouped corrections: show as numbered
        # chain with header. The agent sees the full
        # progression and decides for itself whether
        # corrections conflict or are additive.
        if chain_len >= 2:
            # Emit chain header on first item of chain
            if chain_pos == 1:
                header = "[CORRECTION CHAIN]"
                if subject_str:
                    header += f" for {subject_str}"
                header += f" ({chain_len} corrections, oldest first)"
                lines.append(header)

            tag = " (latest)" if is_latest else ""
            line = f"  {chain_pos}. "
            if correction:
                line += f'"{correction}"'
            line += tag
            lines.append(line)

            # Include content only when this step's
            # correction is ambiguous (references "that",
            # "this", "the last", etc.) and needs the
            # full trace context to be understood. Self-
            # contained corrections stay lean so the
            # chain signal isn't diluted by noise.
            step_needs_ctx = p.get(
                "document_metadata",
                {},
            ).get("correction_needs_context", False)
            if (mode == "raw" and content) or (mode == "adaptive" and content and step_needs_ctx):
                lines.append(
                    f"     [{content}]",
                )

            # Blank line after last chain item
            if chain_pos == chain_len:
                lines.append("")
            continue

        # Non-chain results: regular format
        if aftermath == "corrected":
            # Feature 1: annotate actor ownership so
            # the agent can prefer its own corrections.
            if actor_id:
                doc_actor = doc.get("actor_id")
                if doc_actor == actor_id:
                    tag = "[CORRECTED — your correction]"
                else:
                    tag = "[CORRECTED — other user]"
            else:
                tag = "[CORRECTED]"
        elif aftermath == "self_corrected":
            tag = "[SELF-ASSESSED]"
        elif aftermath == "accepted":
            tag = "[accepted]"
        else:
            tag = f"[{aftermath}]"

        header = f"{tag} {label}"
        if subject_str:
            header += f" for {subject_str}"
        lines.append(header)

        if mode == "raw":
            if content:
                lines.append(f"  Content: {content}")
            if correction:
                lines.append(
                    f'  Correction: "{correction}"',
                )
            if corrected:
                lines.append(
                    f"  Corrected outcome: {corrected}",
                )
        elif mode == "adaptive":
            if correction and not needs_ctx:
                lines.append(
                    f'  Correction: "{correction}"',
                )
            elif correction and needs_ctx:
                if content:
                    lines.append(
                        f"  Content: {content}",
                    )
                lines.append(
                    f'  Correction: "{correction}"',
                )
                if corrected:
                    lines.append(
                        f"  Corrected outcome: {corrected}",
                    )
            else:
                if content:
                    lines.append(
                        f"  Content: {content}",
                    )
        else:
            if correction:
                lines.append(
                    f'  Correction: "{correction}"',
                )

        lines.append("")

    return "\n".join(lines).rstrip()


def format_precedents_naive(result: dict) -> str:
    """Format precedents as a plain content dump.

    No tags, no structured fields — just chunk content
    separated by blank lines.

    Returns an empty string when there are no results.
    """
    precedents = result.get("results", [])
    if not precedents:
        return ""

    lines = ["## Past Runs", ""]
    for p in precedents:
        content = p.get("content", "")
        if content:
            lines.append(content)
            lines.append("")

    return "\n".join(lines).rstrip()
