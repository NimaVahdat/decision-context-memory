"""Minimal runnable demo for the doc_naive approach.

Requires the Docker Postgres DB (docker compose up -d) and OPENAI_API_KEY.
"""

import asyncio

from doc_naive import build


async def main() -> None:
    capture, query = build()

    await capture.initialize()

    trace = {
        "skill_name": "demo_skill",
        "actor_id": "agent_1",
        "subjects": ["acme_corp"],
        "outcome_summary": "Approved the deal after review.",
        "message_history": [
            {"role": "user", "content": "Should we approve the acme_corp deal?"},
            {"role": "assistant", "content": "Reviewing the terms now."},
            {"role": "assistant", "content": "Terms look good; approving."},
        ],
    }
    await capture.capture(trace)

    await query.initialize()
    result = await query.precedents(
        description="approving an acme_corp deal",
        subjects=["acme_corp"],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
