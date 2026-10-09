import numpy as np

from core.vector_store import AutoFaissIndex, VectorStore


def _embeddings(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(rng.random((n, dim)), dtype="float32")


def test_auto_index_selects_flat_for_small_input():
    index = AutoFaissIndex(dimension=8)
    index.select_index_type(3)

    assert index.index_type == "FlatL2"
    assert index.get_index_info()["size"] == 0


def test_auto_index_selects_ivf_variants_by_thresholds():
    medium = AutoFaissIndex(dimension=8)
    medium.small_dataset_threshold = 2
    medium.select_index_type(5)
    assert medium.index_type == "IVFFlat"

    large = AutoFaissIndex(dimension=8)
    large.small_dataset_threshold = 2
    large.medium_dataset_threshold = 2
    large.select_index_type(5)
    assert large.index_type == "IVFPQ"


def test_vector_store_build_search_and_clear_roundtrip():
    store = VectorStore()
    chunks = ["alpha", "beta", "gamma"]
    ids = ["c1", "c2", "c3"]
    metas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]
    vectors = _embeddings(3)

    store.build_index(chunks, ids, metas, vectors)
    assert store.is_ready is True
    assert store.total_chunks == 3

    docs, doc_ids, metadatas = store.search(vectors[0:1], k=3)
    assert doc_ids[0] == "c1"
    assert set(doc_ids) == {"c1", "c2", "c3"}
    assert docs[0] == "alpha"
    assert metadatas[0] == {"source": "a"}

    store.clear()
    assert store.is_ready is False
    assert store.total_chunks == 0
    assert store.contents_map == {}


def test_vector_store_search_on_empty_index_returns_empty():
    store = VectorStore()

    assert store.search(_embeddings(1), k=5) == ([], [], [])
