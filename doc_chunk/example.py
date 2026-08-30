"""Minimal runnable demo for the doc_chunk approach.

Requires the Docker Postgres DB (docker compose up -d) and
OPENAI_API_KEY to be exported in the environment.
"""

import asyncio

from doc_chunk import build


async def main() -> None:
    capture, query = build()

    await capture.initialize()

    trace = {
        "skill_name": "code-review",
        "actor_id": "alice",
        "subjects": ["pr-42"],
        "outcome_summary": "Approved the PR after checking tests.",
        "message_history": [
            {"role": "user", "content": "Review PR #42."},
            {"role": "assistant", "content": "Tests pass; approving."},
        ],
    }
    await capture.capture(trace)

    await query.initialize()
    result = await query.precedents(
        description="Should I approve this pull request?",
        subjects=["pr-42"],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
