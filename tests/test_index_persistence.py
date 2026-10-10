"""P2.1 索引持久化测试。

覆盖：FAISS/BM25 保存-重载往返、重启后恢复、文件缺失未就绪、索引损坏不静默加载、
manifest 字段、配置不匹配拒绝加载。全部使用 ``tmp_path`` 隔离，不触碰项目 data/index/。
"""

import json
import os

import numpy as np
import pytest

import core.bm25_index as bm25_module
import core.vector_store as vector_store_module
from core.bm25_index import BM25IndexManager, BM25_FILENAME
from core.vector_store import (
    VectorStore,
    IndexConfigMismatchError,
    save_persisted_index,
    load_persisted_index,
    FAISS_FILENAME,
    CHUNKS_FILENAME,
    MANIFEST_FILENAME,
)


CHUNKS = ["液压泵异响排查", "空调不制冷原因", "行走马达无力"]
IDS = ["c1", "c2", "c3"]
METAS = [
    {"source": "a.pdf", "doc_id": "d1"},
    {"source": "b.pdf", "doc_id": "d2"},
    {"source": "c.pdf", "doc_id": "d3"},
]


def _embeddings(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(rng.random((n, dim)), dtype="float32")


def _build_store():
    store = VectorStore()
    store.build_index(CHUNKS, IDS, METAS, _embeddings(len(CHUNKS)))
    return store


def _build_bm25():
    manager = BM25IndexManager()
    manager.build_index(CHUNKS, IDS)
    return manager


def _persist_complete(tmp_path, monkeypatch):
    """借助编排函数在 tmp_path 生成一套完整、合法的持久化索引。"""
    monkeypatch.setattr(vector_store_module, "vector_store", _build_store())
    monkeypatch.setattr(bm25_module, "bm25_manager", _build_bm25())
    assert save_persisted_index(index_dir=str(tmp_path)) is True


def test_vector_store_persist_and_reload_roundtrip(tmp_path):
    store = _build_store()
    assert store.save(str(tmp_path)) is True
    assert os.path.exists(tmp_path / FAISS_FILENAME)
    assert os.path.exists(tmp_path / CHUNKS_FILENAME)

    reloaded = VectorStore()
    assert reloaded.load(str(tmp_path)) is True
    assert reloaded.is_ready is True
    assert reloaded.total_chunks == len(CHUNKS)
    assert reloaded.contents_map == store.contents_map
    assert reloaded.metadatas_map == store.metadatas_map
    assert reloaded.id_order == store.id_order

    query = _embeddings(1)
    assert reloaded.search(query, k=3)[1] == store.search(query, k=3)[1]


def test_bm25_persist_and_reload_rebuilds_index(tmp_path):
    manager = _build_bm25()
    assert manager.save(str(tmp_path)) is True

    bm25_path = tmp_path / BM25_FILENAME
    # 必须为 JSON（非 pickle 二进制）
    assert bm25_path.read_bytes()[:1] == b"{"
    payload = json.loads(bm25_path.read_text(encoding="utf-8"))
    assert set(payload) == {"version", "ids", "tokenized"}
    assert payload["ids"] == IDS

    reloaded = BM25IndexManager()
    assert reloaded.load(str(tmp_path), contents_by_id=dict(zip(IDS, CHUNKS))) is True
    assert reloaded.bm25_index is not None
    results = reloaded.search("空调不制冷", top_k=2)
    assert results and results[0]["id"] == "c2"
    assert results[0]["content"] == CHUNKS[1]


def test_persisted_index_restores_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store_module, "vector_store", _build_store())
    monkeypatch.setattr(bm25_module, "bm25_manager", _build_bm25())
    assert save_persisted_index(index_dir=str(tmp_path)) is True

    # 模拟进程重启：换成全新空实例
    monkeypatch.setattr(vector_store_module, "vector_store", VectorStore())
    monkeypatch.setattr(bm25_module, "bm25_manager", BM25IndexManager())

    state = load_persisted_index(index_dir=str(tmp_path))
    assert state["status"] == "loaded"
    assert vector_store_module.vector_store.is_ready is True
    assert vector_store_module.vector_store.total_chunks == len(CHUNKS)
    assert bm25_module.bm25_manager.bm25_index is not None
    hits = bm25_module.bm25_manager.search("液压泵异响", top_k=1)
    assert hits and hits[0]["id"] == "c1"


def test_manifest_records_required_fields(tmp_path, monkeypatch):
    _persist_complete(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for key in ("embedding_model", "chunk_size", "chunk_overlap", "built_at", "num_chunks"):
        assert key in manifest


def test_saving_empty_index_returns_false(tmp_path):
    assert VectorStore().save(str(tmp_path)) is False
    assert BM25IndexManager().save(str(tmp_path)) is False


def test_missing_index_files_report_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store_module, "vector_store", VectorStore())
    monkeypatch.setattr(bm25_module, "bm25_manager", BM25IndexManager())

    state = load_persisted_index(index_dir=str(tmp_path))
    assert state["status"] == "not_ready"
    assert vector_store_module.vector_store.is_ready is False
    assert bm25_module.bm25_manager.bm25_index is None


def test_corrupt_faiss_file_is_not_silently_loaded(tmp_path, monkeypatch):
    _persist_complete(tmp_path, monkeypatch)
    (tmp_path / FAISS_FILENAME).write_bytes(b"this is not a faiss index")

    monkeypatch.setattr(vector_store_module, "vector_store", VectorStore())
    monkeypatch.setattr(bm25_module, "bm25_manager", BM25IndexManager())
    state = load_persisted_index(index_dir=str(tmp_path))
    assert state["status"] == "not_ready"
    assert vector_store_module.vector_store.is_ready is False


def test_corrupt_manifest_is_not_silently_loaded(tmp_path, monkeypatch):
    _persist_complete(tmp_path, monkeypatch)
    (tmp_path / MANIFEST_FILENAME).write_text("{ not valid json", encoding="utf-8")

    monkeypatch.setattr(vector_store_module, "vector_store", VectorStore())
    monkeypatch.setattr(bm25_module, "bm25_manager", BM25IndexManager())
    state = load_persisted_index(index_dir=str(tmp_path))
    assert state["status"] == "not_ready"
    assert vector_store_module.vector_store.is_ready is False


def test_missing_bm25_file_reports_not_ready(tmp_path, monkeypatch):
    _persist_complete(tmp_path, monkeypatch)
    os.remove(tmp_path / BM25_FILENAME)

    monkeypatch.setattr(vector_store_module, "vector_store", VectorStore())
    monkeypatch.setattr(bm25_module, "bm25_manager", BM25IndexManager())
    state = load_persisted_index(index_dir=str(tmp_path))
    assert state["status"] == "not_ready"
    assert vector_store_module.vector_store.is_ready is False
    assert bm25_module.bm25_manager.bm25_index is None


def test_config_mismatch_refuses_load(tmp_path, monkeypatch):
    _persist_complete(tmp_path, monkeypatch)
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_size"] = manifest["chunk_size"] + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = VectorStore()
    with pytest.raises(IndexConfigMismatchError):
        store.load(str(tmp_path))
    assert store.is_ready is False

    monkeypatch.setattr(vector_store_module, "vector_store", VectorStore())
    monkeypatch.setattr(bm25_module, "bm25_manager", BM25IndexManager())
    state = load_persisted_index(index_dir=str(tmp_path))
    assert state["status"] == "mismatch"
    assert vector_store_module.vector_store.is_ready is False
    assert "重建" in state["message"]


def test_chunk_page_metadata_roundtrips_through_persistence(tmp_path):
    metas = [
        {"source": "a.pdf", "doc_id": "d1", "page": 1, "chunk_index": 0, "ocr_used": False},
        {"source": "a.pdf", "doc_id": "d1", "page": 2, "chunk_index": 1, "ocr_used": True},
        {"source": "a.pdf", "doc_id": "d1", "page": 2, "chunk_index": 2, "ocr_used": True},
    ]
    store = VectorStore()
    store.build_index(CHUNKS, IDS, metas, _embeddings(len(CHUNKS)))
    assert store.save(str(tmp_path)) is True

    reloaded = VectorStore()
    assert reloaded.load(str(tmp_path)) is True

    meta = reloaded.metadatas_map["c2"]
    assert meta["page"] == 2
    assert meta["chunk_index"] == 1
    assert meta["ocr_used"] is True
