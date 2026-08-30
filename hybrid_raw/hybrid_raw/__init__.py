"""hybrid_raw — hybrid graph + document memory.

Captures each agent run as a small graph of decision nodes (with the
tool evidence that informed them) plus the documents and edges that
connect them. At query time it ranks precedents with a BM25 + vector
hybrid search and can surface the raw corrections directly.

    from hybrid_raw import build

    capture, query = build()
    await capture.initialize()
    await capture.capture(trace)

    await query.initialize()
    precedents = await query.precedents(description="...", subjects=["acme.com"])
"""

import os

from hybrid_raw.capture.pipeline import HybridCapture
from hybrid_raw.config import OpenAIConfig
from hybrid_raw.query.search import HybridQuery

DEFAULT_DB_URL = "postgresql+asyncpg://dc:dc@localhost:5432/dc_hybrid_raw"

__all__ = ["DEFAULT_DB_URL", "HybridCapture", "HybridQuery", "build"]


def build(
    db_url: str = DEFAULT_DB_URL,
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", ""),
) -> tuple[HybridCapture, HybridQuery]:
    """Construct the (capture, query) pair for this approach."""
    openai_config = OpenAIConfig(api_key=openai_api_key)
    return (
        HybridCapture(db_url, openai_config),
        HybridQuery(db_url, openai_config),
    )
