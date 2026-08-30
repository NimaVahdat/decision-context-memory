"""Import + construction smoke test.

Verifies the package imports, exposes the standard surface, and that
``build()`` constructs a (capture, query) pair offline (both with and
without the reasoning flag). No database or API connection is opened
until ``initialize()`` is called, so this runs without Postgres or an
API key.
"""

import forge_dc
from forge_dc import build


def test_exports():
    assert callable(build)
    assert forge_dc.DEFAULT_DB_URL.startswith("postgresql")


def test_build_constructs_pair():
    capture, query = build()
    assert capture is not None
    assert query is not None


def test_build_reasoning_variant():
    capture, query = build(emit_reasoning=True)
    assert capture is not None
    assert query is not None
