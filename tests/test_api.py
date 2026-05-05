import pytest
from fastapi.testclient import TestClient

from rag import api as api_module


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    sample = tmp_path / "docs"
    sample.mkdir()
    (sample / "a.txt").write_text(
        "RAG retrieves document chunks before answering questions.", encoding="utf-8"
    )
    monkeypatch.setattr(api_module, "DATA_DIR", sample)
    api_module._index = None
    api_module._retriever = None
    with TestClient(api_module.app) as test_client:
        yield test_client
    api_module._index = None
    api_module._retriever = None


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert int(payload["chunks"]) >= 1


def test_query_returns_citations(client: TestClient):
    response = client.post("/query", json={"question": "What does RAG retrieve?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) >= 1
    assert "Sources:" in data["answer"]
