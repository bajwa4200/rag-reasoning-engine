# RAG Reasoning Engine

FastAPI service that indexes local text documents, retrieves relevant chunks with TF-IDF or hash embeddings, and returns **cited** answers without external API keys.

## Features

- `rag/chunker.py` — overlapping text chunks with provenance
- `rag/embeddings.py` — TF-IDF and feature-hash embeddings
- `rag/index.py` — in-memory vector index
- `rag/retriever.py` — top-k retrieval
- `rag/reasoner.py` — cited answer composition
- `rag/api.py` — `/health`, `/query`, `/reindex` endpoints

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
uvicorn rag.api:app --reload
```

## API

- `GET /health` — service status and chunk count
- `POST /query` — `{"question": "...", "top_k": 4}`
- `POST /reindex` — rebuild index from `data/sample_docs/`

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
