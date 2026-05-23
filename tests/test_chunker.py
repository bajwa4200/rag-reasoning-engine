from rag.chunker import chunk_text


def test_chunk_text_produces_overlapping_chunks():
    text = "word " * 200
    chunks = chunk_text(text, "doc1", chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].source_id == "doc1"
    assert chunks[0].chunk_id.startswith("doc1::")
