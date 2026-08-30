"""Minimal forge_dc demo.

Requires a running Postgres (docker compose up -d) and
OPENAI_API_KEY in the environment. Capture compiles the
correction into a RULE; query retrieves precedents + rules.
"""

import asyncio

from forge_dc import build

TRACE = {
    "skill_name": "research-assistant",
    "skill_version": "1",
    "actor_id": "agent-1",
    "actor_type": "agent",
    "subjects": [{"id": "acme", "name": "Acme Corp"}],
    "conversation_id": "conv-1",
    "outcome_summary": "Looked up Acme Corp funding history.",
    "aftermath_status": "corrected",
    "correction_text": (
        "Always cite the funding round date. Do not report a "
        "valuation without naming the lead investor."
    ),
    "message_history": [
        {"role": "user", "content": "What is Acme Corp's latest funding?"},
        {
            "role": "tool_call",
            "name": "web_search",
            "arguments": {"query": "Acme Corp funding round"},
        },
        {
            "role": "tool_result",
            "content": '{"results_count": 3, "top": "Series B, $40M"}',
        },
        {"role": "assistant", "content": "Acme raised a $40M Series B."},
    ],
}


async def main() -> None:
    capture, query = build()

    await capture.initialize()
    await capture.capture(TRACE)

    await query.initialize()
    result = await query.precedents(
        description="Reporting Acme Corp funding details",
        subjects=["acme"],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
