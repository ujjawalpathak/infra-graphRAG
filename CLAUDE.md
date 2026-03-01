# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Ingest pipeline** (run from project root in order):
```bash
python apps/ingest/fetch.py    # scrape docs → data/raw/*.txt
python apps/ingest/chunk.py    # chunk → data/processed/chunks.jsonl
```

**Start API server:**
```bash
uvicorn apps.api.main:app --reload
```

**Run Ollama (required for /chat endpoint):**
```bash
ollama serve
ollama pull llama3.2:1b
```

**Install dependencies:**
```bash
pip install fastapi uvicorn sentence-transformers numpy httpx pydantic python-dotenv pyyaml beautifulsoup4 requests
```

There are no automated tests yet (`tests/` is empty).

## Architecture

The system is a strict RAG pipeline with no fine-tuning or weight modification:

1. **Ingest** (`apps/ingest/`): `fetch.py` scrapes URLs listed in `apps/ingest/sources.yaml`, strips HTML, and saves `.txt` files to `data/raw/`. `chunk.py` splits those into ~1200-char overlapping chunks (200-char overlap) and writes `data/processed/chunks.jsonl` with fields: `chunk_id`, `source_file`, `chunk_index`, `text`.

2. **API** (`apps/api/main.py`): On startup, loads all chunks from `chunks.jsonl` and encodes them into in-memory numpy embeddings using `SentenceTransformer(all-MiniLM-L6-v2)`. Retrieval is pure cosine similarity — no vector DB, no approximate search.

3. **LLM client** (`libs/llm.py`): Thin async wrapper around Ollama's `/api/generate` endpoint. The API layer imports it optionally; `/chat` degrades gracefully if it's unavailable.

**Three endpoints:**
- `POST /search` — returns top-k chunks with scores and text previews
- `POST /ask` — deterministic (no LLM) tutor answer; ranks sentences by keyword overlap with the query
- `POST /chat` — LLM answer via Ollama, gated by `CHAT_MIN_TOP_SCORE` (default 0.55); refuses if top retrieval score is below threshold

**Grounding rules enforced in prompt:** answer only from provided evidence chunks, refuse with "I don't know based on the provided docs." if evidence is insufficient, add inline citations (C1, C2, …).

## Key Constraints

- Retrieval is always cosine similarity — do not replace with probabilistic ranking or approximate methods without discussion.
- The prompt constraint ("answer only from this context") and citation markers (C1, C2, …) must be preserved in all LLM-facing prompts.
- `CHAT_MIN_TOP_SCORE` is the confidence gate — lowering it risks ungrounded answers.
- All data stays local; `libs/llm.py` always points to `127.0.0.1`.

## Environment Variables (`.env`)

Copy `.env.example` to `.env`. Key vars:

| Variable | Default | Purpose |
|---|---|---|
| `CHUNKS_PATH` | `data/processed/chunks.jsonl` | Path to chunked data |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHAT_MIN_TOP_SCORE` | `0.55` | Minimum cosine score to allow LLM generation |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.1` | Model pulled in Ollama |

`POSTGRES_URL` and `NEO4J_*` vars are placeholders for planned Phase 14 (pgvector) and Phase 12 (GraphRAG/Knowledge Graph) work — not yet used.

## Roadmap Context

The project evolves in phases (see `PROJECT_PHASES.md`). Next planned phases:
- **Phase 10**: Hybrid retrieval (BM25 + vector), reranking, query rewriting
- **Phase 11**: Evaluation harness (retrieval accuracy, citation correctness, refusal correctness)
- **Phase 12**: Knowledge Graph integration (GraphRAG) with Neo4j — nodes: Concept, Resource, Configuration, DocPage; edges: DEPENDS_ON, REQUIRES, CONFIGURES, RELATED_TO
- **Phase 13**: Corpus expansion (Terraform, AWS, CNCF docs)
