"""In-memory vector index over document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag.chunker import Chunk, chunk_text
from rag.embeddings import BaseEmbedding, TfidfEmbedding


@dataclass
class IndexedChunk:
    chunk: Chunk
    vector: list[float]


class DocumentIndex:
    """Build and search a local embedding index."""

    def __init__(self, embedder: BaseEmbedding | None = None) -> None:
        self.embedder = embedder or TfidfEmbedding()
        self._entries: list[IndexedChunk] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def add_text(self, text: str, source_id: str) -> int:
        chunks = chunk_text(text, source_id)
        for chunk in chunks:
            self._entries.append(IndexedChunk(chunk=chunk, vector=[]))
        return len(chunks)

    def add_directory(self, directory: Path, pattern: str = "*.txt") -> int:
        total = 0
        for path in sorted(directory.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            total += self.add_text(text, source_id=path.stem)
        return total

    def build(self) -> None:
        documents = [e.chunk.text for e in self._entries]
        self.embedder.fit(documents)
        for entry in self._entries:
            entry.vector = self.embedder.embed(entry.chunk.text)

    def build_from_directory(self, directory: Path, pattern: str = "*.txt") -> None:
        self.add_directory(directory, pattern=pattern)
        self.build()

    def build_from_texts(self, texts: dict[str, str]) -> None:
        for source_id, text in texts.items():
            self.add_text(text, source_id=source_id)
        self.build()

    def search(self, query: str, top_k: int = 5) -> list[tuple[IndexedChunk, float]]:
        if not self._entries:
            return []
        query_vec = self.embedder.embed(query)
        scored: list[tuple[IndexedChunk, float]] = []
        for entry in self._entries:
            score = self.embedder.cosine_similarity(query_vec, entry.vector)
            scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
