"""Import + construction smoke test.

Verifies the package imports, exposes the standard surface, and that
``build()`` constructs a (capture, query) pair offline. No database or
API connection is opened until ``initialize()`` is called, so this runs
without Postgres or an API key.
"""

import doc_chunk
from doc_chunk import build


def test_exports():
    assert callable(build)
    assert doc_chunk.DEFAULT_DB_URL.startswith("postgresql")


def test_build_constructs_pair():
    capture, query = build()
    assert capture is not None
    assert query is not None
