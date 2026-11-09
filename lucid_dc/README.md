# lucid_dc

Memory that grades itself. It captures each run as a graph of decision nodes,
then runs a **consolidation pass** where an LLM reviews the run's own observable
outcomes and writes back autonomously-learned directives — no human label
required. Those consolidated lessons feed future retrieval.

## How it works

**Capture.** The run is stored as a graph of reasoning/tool decision nodes with
evidence and edges, in Postgres (pgvector).

**Consolidate.** `await capture.consolidate(skill_name=...)` reviews the captured
traces and distills directives from what actually happened (e.g.
"DO NOT call eval_lookup with source='financials'"), stored as consolidated
decision documents.

**Query.** Ranks precedents with hybrid (BM25 + vector) search over the nodes,
returning the consolidated guidance for the next task.

## Quickstart

```bash
docker compose up -d           # Postgres + pgvector on :5432 (db: dc_lucid)
uv pip install -e .            # or: pip install -e .
export OPENAI_API_KEY=sk-...
python example.py
```

```python
import asyncio
from lucid_dc import build

async def main():
    capture, query = build()
    await capture.initialize()
    await capture.capture(trace)
    await capture.consolidate(skill_name="eval-analyst")   # self-assessment pass

    await query.initialize()
    text = await query.precedents(
        description="Analyze acme.com",
        subjects=["acme.com"],
    )
    print(text)

asyncio.run(main())
```

## API

- `build(db_url=DEFAULT_DB_URL, openai_api_key=os.environ["OPENAI_API_KEY"])`
  → `(LucidCapture, LucidQuery)`.
- `LucidCapture`: `initialize()`, `capture(trace)`, `consolidate(skill_name=…)`,
  `shutdown()`.
- `LucidQuery`: `initialize()`, `precedents(description, subjects=…, limit=5)`,
  `search(...)`, `global_readout(skill_name, …)`, `shutdown()`.

## Requirements

- Python ≥ 3.12, `OPENAI_API_KEY` (embeddings + a strong model for consolidation).
- Postgres with the `vector` extension (bundled `docker-compose.yml`). Tables are
  created on `initialize()`.

## Citation

`lucid_dc` is stage 4 of the five-stage progression in **[Decision Context: Persistent Behavioral Learning for
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
