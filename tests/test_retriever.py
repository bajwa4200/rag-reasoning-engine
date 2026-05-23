from pathlib import Path

from rag.index import DocumentIndex
from rag.retriever import Retriever

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_retrieve_returns_scored_chunks():
    index = DocumentIndex()
    index.build_from_directory(DATA)
    retriever = Retriever(index)
    hits = retriever.retrieve("TF-IDF cosine similarity", top_k=2)
    assert hits
    assert hits[0].score > 0
    assert "[" in hits[0].citation
