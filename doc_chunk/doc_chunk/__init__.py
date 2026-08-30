"""doc_chunk — chunked document Decision Context memory approach.

Captures agent traces as a document plus tool-call-grouped,
annotated chunks (no graph, no edges). Query uses BM25 + vector
hybrid search over chunks and returns raw precedent dumps.
"""

import os

from doc_chunk.capture import DocCapture
from doc_chunk.config import OpenAIConfig
from doc_chunk.query import DocQuery

DEFAULT_DB_URL = "postgresql+asyncpg://dc:dc@localhost:5432/dc_doc_chunk"

__all__ = ["DEFAULT_DB_URL", "DocCapture", "DocQuery", "build"]


def build(
    db_url: str = DEFAULT_DB_URL,
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", ""),
) -> tuple[DocCapture, DocQuery]:
    """Construct the (capture, query) pair for this approach."""
    openai_config = OpenAIConfig(api_key=openai_api_key)
    capture = DocCapture(db_url, openai_config)
    query = DocQuery(db_url, openai_config)
    return capture, query
