"""Split documents into overlapping text chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A slice of source text with provenance metadata."""

    text: str
    source_id: str
    chunk_id: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source_id: str,
    *,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Split *text* into fixed-size chunks with character overlap."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(1, chunk_size - overlap)

    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        piece = normalized[start:end]
        chunks.append(
            Chunk(
                text=piece,
                source_id=source_id,
                chunk_id=f"{source_id}::{index}",
                start_char=start,
                end_char=end,
            )
        )
        if end >= len(normalized):
            break
        start += step
        index += 1

    return chunks
