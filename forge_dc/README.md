# forge_dc

Memory that compiles corrections into rules. Instead of replaying past prose,
a strong LLM turns the human corrections from past runs into strict, reusable
`RULE:` directives. At query time the agent receives only those terse,
authoritative rules — not raw transcripts.

## How it works

**Capture.** Each run is stored as a graph of decision nodes (with evidence and
edges) in Postgres (pgvector).

**Consolidate.** `await capture.consolidate(skill_name=...)` walks the correction
chains and a strong LLM compiles them into `RULE:` directives, stored as decision
nodes wired back to their source runs via `FORGED_FROM` edges.

**Query.** Hybrid (BM25 + vector) retrieval finds the relevant decision nodes,
follows the `FORGED_FROM` edges to the compiled rules, and injects **only the
rules**.

### Reasoning variant

`build(emit_reasoning=True)` makes the compiler also attach a one-line **WHY**
rationale to each rule. The default (`emit_reasoning=False`) emits rules only.

## Quickstart

```bash
docker compose up -d           # Postgres + pgvector on :5432 (db: dc_forge)
uv pip install -e .            # or: pip install -e .
export OPENAI_API_KEY=sk-...
python example.py
```

```python
import asyncio
from forge_dc import build

async def main():
    capture, query = build()                       # rules only
    # capture, query = build(emit_reasoning=True)   # rules + WHY
    await capture.initialize()
    await capture.capture(trace)
    await capture.consolidate(skill_name="eval-analyst")   # compile rules

    await query.initialize()
    text = await query.precedents(
        description="Analyze acme.com",
        subjects=["acme.com"],
    )
    print(text)                                    # "# MANDATORY RULES — ..."

asyncio.run(main())
```

## API

- `build(db_url=DEFAULT_DB_URL, openai_api_key=os.environ["OPENAI_API_KEY"], emit_reasoning=False)`
  → `(ForgeCapture, ForgeQuery)`.
- `ForgeCapture`: `initialize()`, `capture(trace)`, `consolidate(skill_name=…)`,
  `shutdown()`.
- `ForgeQuery`: `initialize()`, `precedents(description, subjects=…, limit=5)`,
  `evidence_corrections(...)`, `shutdown()`.

## Requirements

- Python ≥ 3.12, `OPENAI_API_KEY` (embeddings + a strong model for compilation).
- Postgres with the `vector` extension (bundled `docker-compose.yml`). Tables are
  created on `initialize()`.

## Citation

`forge_dc` is the final stage of the five-stage progression and the paper’s headline system in **[Decision Context: Persistent Behavioral Learning for
AI Agents](https://arxiv.org/abs/2611.08644)** (NeurIPS 2026 PALM workshop). If
you use it, please cite:

```bibtex
@misc{vahdat2026decisioncontext,
  title         = {Decision Context: Persistent Behavioral Learning for AI Agents},
  author        = {Nima Vahdat and Pashootan Vaezipoor},
  year          = {2026},
  eprint        = {2611.08644},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  note          = {NeurIPS 2026 Workshop on Personalized, Aligned, Long-Term Memory for AI Systems (PALM)},
  url           = {https://arxiv.org/abs/2611.08644}
}
```

- Paper: [arXiv:2611.08644](https://arxiv.org/abs/2611.08644)
- Workshop version: [OpenReview](https://openreview.net/forum?id=7XhMLOsQoA)
  (NeurIPS 2026 PALM workshop, non-archival)
- All five approaches: [`decision-context-memory`](https://github.com/NimaVahdat/decision-context-memory)
- Benchmark: [`decision-context-benchmark`](https://github.com/NimaVahdat/decision-context-benchmark)
