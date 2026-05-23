from pathlib import Path

from rag.index import DocumentIndex

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def test_build_from_directory():
    index = DocumentIndex()
    index.build_from_directory(DATA)
    assert index.size > 0


def test_build_from_texts():
    index = DocumentIndex()
    index.build_from_texts({"a": "hello world", "b": "foo bar baz"})
    assert index.size >= 2
