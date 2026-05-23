from rag.embeddings import HashEmbedding, TfidfEmbedding


def test_tfidf_similarity_for_related_text():
    embedder = TfidfEmbedding()
    docs = ["cats like fish", "dogs like bones", "cats enjoy fish"]
    embedder.fit(docs)
    a = embedder.embed("cats fish")
    b = embedder.embed("cats enjoy fish")
    c = embedder.embed("rockets launch")
    assert embedder.cosine_similarity(a, b) > embedder.cosine_similarity(a, c)


def test_hash_embedding_is_normalized():
    embedder = HashEmbedding(dim=64)
    embedder.fit(["hello world"])
    vec = embedder.embed("hello world")
    assert len(vec) == 64
    assert 0.99 < sum(v * v for v in vec) ** 0.5 <= 1.01
