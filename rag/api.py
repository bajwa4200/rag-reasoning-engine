"""FastAPI application for the RAG reasoning engine."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.index import DocumentIndex
from rag.reasoner import Reasoner
from rag.retriever import Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

app = FastAPI(title="RAG Reasoning Engine", version="0.1.0")

_index: DocumentIndex | None = None
_retriever: Retriever | None = None
_reasoner = Reasoner()


def _ensure_index() -> tuple[DocumentIndex, Retriever]:
    global _index, _retriever
    if _index is None:
        _index = DocumentIndex()
        if DATA_DIR.is_dir():
            _index.add_directory(DATA_DIR)
        _index.build()
        _retriever = Retriever(_index)
    assert _retriever is not None
    return _index, _retriever


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, str | float]]
    context: str


@app.get("/health")
def health() -> dict[str, str]:
    index, _ = _ensure_index()
    return {"status": "ok", "chunks": str(index.size)}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    _, retriever = _ensure_index()
    retriever.top_k = body.top_k
    results = retriever.retrieve(body.question)
    cited = _reasoner.answer(body.question, results)
    return QueryResponse(
        answer=cited.answer,
        citations=cited.citations,
        context=Retriever.format_context(results),
    )


@app.post("/reindex")
def reindex() -> dict[str, int]:
    global _index, _retriever
    _index = DocumentIndex()
    count = _index.add_directory(DATA_DIR) if DATA_DIR.is_dir() else 0
    _index.build()
    _retriever = Retriever(_index)
    return {"chunks": _index.size, "documents": count}
