# Decision Context Memory

Behavioral memory for LLM agents. Each approach here lets an agent **learn from
past corrections** — capture what happened on previous runs (and how a human
corrected it), then surface the right guidance before the next run so the agent
stops repeating the same mistakes.

Every approach is a **fully self-contained folder**: its own dependencies, its
own database, its own `docker-compose.yml`, its own README and runnable example.
Nothing is shared between folders — copy a single folder out of this repo and it
works on its own.

## The approaches

| Folder | One line | Storage |
|---|---|---|
| [`doc_naive`](doc_naive/) | Store each run as embedded chunks; retrieve the most similar past chunks as a plain text dump. | Postgres + pgvector |
| [`doc_chunk`](doc_chunk/) | Same capture, but rank chunks with BM25 + vector hybrid search and return structured precedents. | Postgres + pgvector |
| [`hybrid_raw`](hybrid_raw/) | Store each run as a small graph of decision nodes, evidence, and edges; retrieve precedents and surface the raw corrections. | Postgres + pgvector |
| [`lucid_dc`](lucid_dc/) | Hybrid capture plus a consolidation pass where an LLM reviews the run's own outcomes and writes autonomously-learned directives. | Postgres + pgvector |
| [`forge_dc`](forge_dc/) | Hybrid capture plus rule compilation: a strong LLM turns corrections into strict, reusable `RULE` directives and injects only those. | Postgres + pgvector |

(The paper also benchmarks two external products, `mem0` and `Zep`. They are not
part of this repo because they aren't ours.)

## Common shape

Every folder exposes the same small surface:

```python
from <approach> import build

capture, query = build()          # uses OPENAI_API_KEY + a local Postgres
await capture.initialize()        # creates tables on first use
await capture.capture(trace)      # ingest one agent run (+ any correction)
# lucid_dc / forge_dc also: await capture.consolidate(skill_name="...")

await query.initialize()
precedents = await query.precedents(description="...", subjects=["acme.com"])
# `precedents` is the text you inject into the agent's next run
```

- **`build(db_url=..., openai_api_key=...)`** returns a `(capture, query)` pair.
- **Capture** ingests an agent run trace (messages + tool calls + outcome, plus a
  human correction if there was one).
- **Query** returns the precedents/rules relevant to the next task.

## Using one approach

Each folder is independent. For example:

```bash
cd forge_dc
docker compose up -d              # start its Postgres
uv pip install -e .               # or: pip install -e .
export OPENAI_API_KEY=sk-...
python example.py
```

See each folder's README for the details and the exact API it exposes.

## Requirements

- Python ≥ 3.12
- An `OPENAI_API_KEY` (embeddings; plus a strong model for `lucid_dc` / `forge_dc`)
- Docker (each folder ships a `docker-compose.yml` for its database)

Each folder pins only the dependencies it actually needs — none require Neo4j.

## Development

A root `Makefile` runs the same checks CI does, looping over every approach
folder. Scope to one folder with `APPROACHES=<folder>`.

```bash
make install         # uv sync each folder
make lint-check      # decoupling guard + ruff over every folder
make typing          # mypy over every folder
make test            # pytest (import/build smoke) over every folder
make clean           # remove caches and build artifacts

make lint-check APPROACHES=forge_dc   # just one folder
```

`make decouple-check` (also run by `lint-check`) enforces the core contract:
no folder imports the legacy `dc_approaches` package or any other approach's
package, so each folder stays copy-and-go. CI (`.github/workflows/ci.yml`) runs
the decoupling guard once and then lints, type-checks, and tests each folder as
an independent matrix job.
