"""Aftermath classification — correction vs new request.

Deterministic heuristic first (fast, free). Falls back
to a cheap LLM call only when uncertain.
"""

import logging
from typing import NamedTuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class FollowupClassification(NamedTuple):
    """Result of classify_followup."""

    classification: str  # "correction" or "new_request"
    needs_context: bool  # True if correction is ambiguous


_CORRECTION_SIGNALS = [
    # Explicit correction language
    "wrong",
    "fix",
    "correct",
    "not right",
    "redo",
    "revise",
    "actually",
    "instead",
    # Modification instructions
    "change",
    "update",
    "convert",
    "replace",
    "swap",
    # Addition/removal
    "include",
    "add",
    "also",
    "remove",
    "drop",
    # Missing content
    "missing",
    "forgot",
    # Formatting instructions
    "bold",
    "number ",
    "format",
    "present ",
    "show ",
    "put ",
    "make ",
    "list ",
    # Rules / constraints
    "always",
    "never",
    "don't",
    "do not",
    "stop",
    "keep to",
    "should have",
    "should be",
    "must ",
    # Specific patterns
    "use ",
    "lead with",
    "end with",
    "start with",
    "express",
    "cap ",
]

# Signals that suggest the correction references the
# output and can't be understood standalone.
_CONTEXT_SIGNALS = [
    "that ",
    "this ",
    "the last",
    "the first",
    "above",
    "below",
    "previous",
    "it ",
    "them ",
    "those ",
    "these ",
]


def classify_followup_deterministic(
    human_message: str,
) -> str | None:
    """Deterministic correction classification.

    Returns "correction" if signal words found, None
    if uncertain (caller falls back to LLM).
    """
    msg_lower = human_message.lower()
    for signal in _CORRECTION_SIGNALS:
        if signal in msg_lower:
            return "correction"
    return None


def _needs_context_deterministic(
    human_message: str,
) -> bool:
    """Determine if a correction needs context.

    Returns True if the message uses references that
    only make sense with the original output visible
    (e.g. "remove that last section").
    """
    msg_lower = human_message.lower()
    return any(signal in msg_lower for signal in _CONTEXT_SIGNALS)


async def classify_followup(
    client: AsyncOpenAI,
    human_message: str,
    prev_outcome: str,
    prev_skill: str = "",
    new_skill: str = "",
) -> FollowupClassification:
    """Classify follow-up via LLM.

    Uses gpt-4.1-mini for speed and cost. Truncates
    the previous outcome to keep prompt small.
    Single question: CORRECTION or NEW_REQUEST.
    Context classification done deterministically.
    """
    # Truncate outcome — the LLM only needs a summary
    # to understand what the agent produced, not the
    # full output.
    outcome_short = prev_outcome[:200] if prev_outcome else "(no output)"

    prompt = (
        "A user sent a follow-up after an AI agent "
        "produced output. Is the follow-up a "
        "CORRECTION (wants to change/fix/adjust the "
        "output) or a NEW_REQUEST (asking something "
        "new)?\n\n"
        f"Agent output (truncated): {outcome_short}\n"
        f"User follow-up: {human_message}\n\n"
        'Answer: {"type": "CORRECTION"} or '
        '{"type": "NEW_REQUEST"}'
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": ("Classify the message. Respond with JSON only. No explanation."),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=50,
            temperature=0,
        )

        text = response.choices[0].message.content or ""
        logger.info(
            "classify_followup: msg=%r -> %r",
            human_message[:60],
            text.strip(),
        )

        upper = text.strip().upper()

        if not upper:
            logger.warning("classify_followup: empty response")
            return FollowupClassification(
                classification="correction",
                needs_context=True,
            )

        is_correction = "CORRECTION" in upper
        classification = "correction" if is_correction else "new_request"

        # Context classification is deterministic —
        # no need for a second LLM call.
        needs_ctx = _needs_context_deterministic(human_message) if is_correction else False

        return FollowupClassification(
            classification=classification,
            needs_context=needs_ctx,
        )

    except Exception as exc:
        logger.warning("classify_followup failed: %s", exc)
        # Default to correction — safer than losing
        # a correction
        return FollowupClassification(
            classification="correction",
            needs_context=True,
        )
