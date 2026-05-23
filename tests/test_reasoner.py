from rag.reasoner import Reasoner
from rag.retriever import RetrievalResult


def test_reasoner_includes_citations():
    results = [
        RetrievalResult(
            chunk_id="doc::0",
            source_id="doc",
            text="RAG combines retrieval with generation for grounded answers.",
            score=0.8,
        )
    ]
    cited = Reasoner().answer("What is RAG?", results)
    assert "[1]" in cited.answer
    assert cited.citations[0]["source_id"] == "doc"
