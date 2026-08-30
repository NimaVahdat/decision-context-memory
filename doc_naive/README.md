# doc_naive

The simplest memory: store, embed, retrieve. No structure, no analysis, no rule
extraction — just the raw text of past runs, surfaced by similarity. It's the
honest baseline you compare everything else against.

## How it works

**Capture.** Each agent run is stored as a *document*, split into chunks at
tool-call boundaries, and every chunk is embedded and written to Postgres
(pgvector). Nothing is summarized or interpreted.

**Query.** Given a description of the next task, it embeds the query, finds the
most similar past chunks by vector similarity, and returns them as a **plain
text dump** (no tags, no formatting) for you to prepend to the agent's prompt.

## Quickstart

```bash
docker compose up -d           # Postgres + pgvector on :5432 (db: dc_doc_naive)
uv pip install -e .            # or: pip install -e .
export OPENAI_API_KEY=sk-...
python example.py
```

```python
import asyncio
from doc_naive import build

async def main():
    capture, query = build()                 # DEFAULT_DB_URL + OPENAI_API_KEY
    await capture.initialize()                # creates tables on first run
    await capture.capture(trace)              # ingest one agent run

    await query.initialize()
    text = await query.precedents(
        description="Analyze acme.com",
        subjects=["acme.com"],
    )
    print(text)                               # inject this into the next run

asyncio.run(main())
```

## API

- `build(db_url=DEFAULT_DB_URL, openai_api_key=os.environ["OPENAI_API_KEY"])`
  → `(DocNaiveCapture, DocQuery)`.
- `DocNaiveCapture`: `initialize()`, `capture(trace) -> result`, `shutdown()`.
- `DocQuery` (naive mode): `initialize()`, `precedents(description, subjects=…, limit=5)`,
  `search(...)`, `shutdown()`.

## Requirements

- Python ≥ 3.12, `OPENAI_API_KEY` (embeddings).
- Postgres with the `vector` extension — the bundled `docker-compose.yml` provides it.
  Tables are created automatically on `initialize()`.
