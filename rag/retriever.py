"""Retrieve relevant chunks for a user query."""

from __future__ import annotations

from dataclasses import dataclass

from rag.index import DocumentIndex, IndexedChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    source_id: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.source_id}#{self.chunk_id}]"


class Retriever:
    def __init__(self, index: DocumentIndex, top_k: int = 4) -> None:
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k if top_k is not None else self.top_k
        hits = self.index.search(query, top_k=k)
        return [
            RetrievalResult(
                chunk_id=entry.chunk.chunk_id,
                source_id=entry.chunk.source_id,
                text=entry.chunk.text,
                score=score,
            )
            for entry, score in hits
            if score > 0
        ]

    @staticmethod
    def format_context(results: list[RetrievalResult]) -> str:
        lines: list[str] = []
        for i, result in enumerate(results, start=1):
            lines.append(
                f"[{i}] source={result.source_id} chunk={result.chunk_id}\n{result.text}"
            )
        return "\n\n".join(lines)
