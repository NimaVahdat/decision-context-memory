"""Minimal runnable demo for the hybrid_raw approach.

Requires a running Postgres+pgvector (docker compose up -d) and the
OPENAI_API_KEY environment variable to be set.
"""

import asyncio

from hybrid_raw import build

TRACE = {
    "skill_name": "refund-handler",
    "skill_version": "1.0",
    "actor_id": "agent-1",
    "actor_type": "agent",
    "subjects": ["order-42"],
    "outcome_summary": "Approved a refund for a delayed order.",
    "message_history": [
        {"role": "user", "content": "Order 42 arrived 3 weeks late. Refund?"},
        {
            "role": "assistant",
            "content": "Policy allows refunds for delays over 2 weeks. Approving.",
        },
    ],
}


async def main() -> None:
    capture, query = build()
    await capture.initialize()
    await capture.capture(TRACE)

    await query.initialize()
    result = await query.precedents(
        description="customer wants a refund for a late delivery",
        subjects=["order-42"],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
