"""Minimal runnable demo for the lucid_dc approach.

Requires a running Postgres (docker compose up -d) and OPENAI_API_KEY
to be set in the environment (embeddings are computed via OpenAI).
"""

import asyncio

from lucid_dc import build

TRACE = {
    "skill_name": "refund-handler",
    "actor_id": "agent-1",
    "subjects": ["order-42"],
    "outcome_summary": "Approved a refund for a delayed order.",
    "message_history": [
        {
            "role": "user",
            "content": "Customer wants a refund for order-42, it arrived 9 days late.",
        },
        {
            "role": "assistant",
            "content": "Policy allows full refund when delivery is >7 days late. Issuing refund.",
        },
        {"role": "tool", "content": "refund_issued(order-42, amount=49.99)"},
    ],
}


async def main():
    capture, query = build()
    await capture.initialize()
    await capture.capture(TRACE)

    await query.initialize()
    result = await query.precedents(
        description="Should I refund an order that shipped late?",
        subjects=["order-42"],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
