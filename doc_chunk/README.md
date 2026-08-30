# doc_chunk

Document memory with a smarter retriever. It captures the same chunked documents
as a plain store, but each chunk is annotated and retrieval uses **BM25 + vector
hybrid search**, returning structured precedents instead of a raw dump.

## How it works

**Capture.** Each agent run is stored as a *document* split into
tool-call-grouped chunks. Each chunk is annotated (action label, tool name, a
short evidence/outcome summary) and embedded into Postgres (pgvector). No graph,
no edges.

**Query.** Given the next task, it scores chunks with a hybrid of BM25 keyword
matching and vector similarity, then formats the top matches as **structured
precedents** (tagged with their action/outcome) for prompt injection.

## Quickstart

```bash
docker compose up -d           # Postgres + pgvector on :5432 (db: dc_doc_chunk)
uv pip install -e .            # or: pip install -e .
export OPENAI_API_KEY=sk-...
python example.py
```

```python
import asyncio
from doc_chunk import build

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
  → `(DocCapture, DocQuery)`.
- `DocCapture`: `initialize()`, `capture(trace)`, `shutdown()`.
- `DocQuery`: `initialize()`, `precedents(description, subjects=…, limit=5)`,
  `search(query_text, search_mode="hybrid", …)`, `shutdown()`.

## Requirements

- Python ≥ 3.12, `OPENAI_API_KEY` (embeddings).
- Postgres with the `vector` extension (bundled `docker-compose.yml`). Tables are
  created on `initialize()`.
