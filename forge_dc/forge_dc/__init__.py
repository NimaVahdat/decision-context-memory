"""forge_dc — rule-compilation memory.

Captures each agent run as a graph of decision nodes, then a strong
LLM compiles the human corrections in those runs into strict, reusable
RULE directives. At query time it retrieves the relevant decision
nodes, follows their FORGED_FROM edges to the compiled rules, and
injects only those rules.

Set ``emit_reasoning=True`` to additionally attach a one-line WHY
rationale to each compiled rule.

    from forge_dc import build

    capture, query = build()                       # rules only
    # capture, query = build(emit_reasoning=True)   # rules + WHY

    await capture.initialize()
    await capture.capture(trace)
    await capture.consolidate(skill_name="...")    # compile rules

    await query.initialize()
    precedents = await query.precedents(description="...", subjects=["acme.com"])
"""

import os

from forge_dc.capture.pipeline import ForgeCapture
from forge_dc.config import OpenAIConfig
from forge_dc.query.search import ForgeQuery

__all__ = ["DEFAULT_DB_URL", "ForgeCapture", "ForgeQuery", "build"]

DEFAULT_DB_URL = "postgresql+asyncpg://dc:dc@localhost:5432/dc_forge"


def build(
    db_url: str = DEFAULT_DB_URL,
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", ""),
    emit_reasoning: bool = False,
):
    """Construct the (capture, query) pair for this approach.

    ``emit_reasoning`` controls whether the rule compiler also emits a
    one-line WHY rationale for each rule.
    """
    os.environ["FORGE_EMIT_WHY"] = "1" if emit_reasoning else "0"
    openai_config = OpenAIConfig(api_key=openai_api_key)
    capture = ForgeCapture(db_url, openai_config)
    query = ForgeQuery(db_url, openai_config)
    return (capture, query)
