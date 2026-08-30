# hybrid_raw

Graph-structured memory with document-store simplicity. Each run becomes a small
graph of decision nodes (plus the tool evidence that informed them and the edges
that connect them), and retrieval ranks precedents with hybrid search — giving
chain traversal and "why did it do that" lookups without a separate graph
database. It surfaces the **raw** corrections it finds.

## How it works

**Capture.** The run's message history is split into *reasoning* and *tool*
nodes, each annotated and embedded (node-level plus an accumulated context
embedding). Nodes are wired into a lightweight graph — singletons (actor /
subject / policy), evidence, and `LED_TO` / `INFORMED_BY` edges — and stored in
Postgres (pgvector).

**Query.** Combines vector + BM25 search over nodes to surface relevant
precedents, with support for walking a decision chain, a skill-level "global
readout," and evidence-linked corrections. Returns the corrections directly.

## Quickstart

```bash
docker compose up -d           # Postgres + pgvector on :5432 (db: dc_hybrid_raw)
uv pip install -e .            # or: pip install -e .
export OPENAI_API_KEY=sk-...
python example.py
```

```python
import asyncio
from hybrid_raw import build

async def main():
    capture, query = build()
    await capture.initialize()
    await capture.capture(trace)

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
  → `(HybridCapture, HybridQuery)`.
- `HybridCapture`: `initialize()`, `capture(trace)`, `shutdown()`.
- `HybridQuery`: `initialize()`, `precedents(description, subjects=…, limit=5)`,
  `search(...)`, `global_readout(skill_name, …)`, `chain(node_id)`, `why(node_id)`,
  `shutdown()`.

## Requirements

- Python ≥ 3.12, `OPENAI_API_KEY` (embeddings).
- Postgres with the `vector` extension (bundled `docker-compose.yml`). Tables are
  created on `initialize()`.
