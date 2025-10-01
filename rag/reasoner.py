"""Compose cited answers from retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.retriever import RetrievalResult


@dataclass(frozen=True)
class CitedAnswer:
    answer: str
    citations: list[dict[str, str | float]]


class Reasoner:
    """Rule-based reasoner that stitches evidence into a cited response."""

    def __init__(self, min_score: float = 0.01) -> None:
        self.min_score = min_score

    def answer(self, query: str, results: list[RetrievalResult]) -> CitedAnswer:
        filtered = [r for r in results if r.score >= self.min_score]
        if not filtered:
            return CitedAnswer(
                answer="I could not find relevant information in the indexed documents.",
                citations=[],
            )

        sentences = self._extract_sentences(query, filtered)
        body = " ".join(sentences) if sentences else filtered[0].text
        citations = [
            {
                "ref": f"[{i}]",
                "source_id": r.source_id,
                "chunk_id": r.chunk_id,
                "score": round(r.score, 4),
            }
            for i, r in enumerate(filtered, start=1)
        ]
        cited_refs = ", ".join(c["ref"] for c in citations)
        answer = f"{body} (Sources: {cited_refs})"
        return CitedAnswer(answer=answer, citations=citations)

    def _extract_sentences(
        self, query: str, results: list[RetrievalResult]
    ) -> list[str]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        chosen: list[str] = []
        seen: set[str] = set()

        for result in results:
            for sentence in re.split(r"(?<=[.!?])\s+", result.text):
                sentence = sentence.strip()
                if len(sentence) < 20 or sentence in seen:
                    continue
                terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(query_terms & terms)
                if overlap > 0:
                    chosen.append(sentence)
                    seen.add(sentence)
                if len(chosen) >= 3:
                    return chosen
        return chosen
