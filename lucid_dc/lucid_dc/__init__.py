"""lucid_dc — Decision Context with self-assessment.

Captures each agent run as a graph of decision nodes and, after
capture, runs a consolidation pass that reviews the run's own
observable outcomes and writes back autonomously-learned
corrections (no human label required). Queries return the
consolidated precedents.

    from lucid_dc import build

    capture, query = build()
    await capture.initialize()
    await capture.capture(trace)
    await capture.consolidate(skill_name="...")   # self-assessment pass

    await query.initialize()
    precedents = await query.precedents(description="...", subjects=["acme.com"])
"""

import os

from lucid_dc.capture.pipeline import LucidCapture
from lucid_dc.config import OpenAIConfig
from lucid_dc.store.query.search import StoreQuery as LucidQuery

DEFAULT_DB_URL = "postgresql+asyncpg://dc:dc@localhost:5432/dc_lucid"

__all__ = ["DEFAULT_DB_URL", "LucidCapture", "LucidQuery", "build"]


def build(
    db_url: str = DEFAULT_DB_URL,
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", ""),
):
    """Construct the (capture, query) pair for this approach."""
    openai_config = OpenAIConfig(api_key=openai_api_key)
    capture = LucidCapture(db_url, openai_config)
    query = LucidQuery(db_url, openai_config)
    return capture, query
