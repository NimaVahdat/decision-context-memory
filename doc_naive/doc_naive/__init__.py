"""doc_naive — naive document memory.

Captures each agent run as a document split into tool-call-grouped
chunks with embeddings. At query time it retrieves the most similar
past chunks by vector similarity and returns them as a plain,
unstructured text dump (no tags, no graph, no rule compilation).

    from doc_naive import build

    capture, query = build()
    await capture.initialize()
    await capture.capture(trace)

    await query.initialize()
    precedents = await query.precedents(description="...", subjects=["acme.com"])
"""

import os

from doc_naive.capture_naive import DocNaiveCapture
from doc_naive.config import OpenAIConfig
from doc_naive.query import DocQuery

DEFAULT_DB_URL = "postgresql+asyncpg://dc:dc@localhost:5432/dc_doc_naive"

__all__ = ["DEFAULT_DB_URL", "DocNaiveCapture", "DocQuery", "OpenAIConfig", "build"]


def build(
    db_url: str = DEFAULT_DB_URL,
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", ""),
) -> tuple[DocNaiveCapture, DocQuery]:
    """Construct the (capture, query) pair for this approach."""
    openai_config = OpenAIConfig(api_key=openai_api_key)
    capture = DocNaiveCapture(db_url, openai_config)
    query = DocQuery(db_url, openai_config, naive=True)
    return (capture, query)
