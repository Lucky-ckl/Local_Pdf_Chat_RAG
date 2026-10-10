"""知识库目录模式（core/ingest.py::sync_documents）与前端隐藏工程信息的验收测试。

全部通过 monkeypatch 隔离：不加载模型、不联网；向量部分使用真实 ``VectorStore``（FAISS）
以验证「追加而不重建」的真实累计行为。
"""

import asyncio
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_router
import core.ingest as ingest
from core.document_loader import Block
from core.text_splitter import Chunk
from core.vector_store import VectorStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeBM25:
    def __init__(self):
        self.calls = []

    def clear(self):
        self.calls.append(("clear",))

    def build_index(self, chunks, ids):
        self.calls.append(("build", len(chunks), len(ids)))


def _embed(texts, show_progress=False):
    # alpha* -> [1,0,0]；其余 -> [0,1,0]，便于用查询向量验证「旧文档仍可检索」。
    return np.array([[1.0, 0.0, 0.0] if t.startswith("alpha") else [0.0, 1.0, 0.0]
                     for t in texts], dtype="float32")


def _load(path, force_ocr=False):
    name = os.path.basename(path)
    text = "alpha-content" if name.startswith("a") else "beta-content"
    return [Block(text=text, page=1, source=name)]


@pytest.fixture
def kb(monkeypatch):
    store = VectorStore()
    bm25 = _FakeBM25()
    clears = {"count": 0}
    original_clear = store.clear

    def counting_clear():
        clears["count"] += 1
        original_clear()

    monkeypatch.setattr(store, "clear", counting_clear)
    monkeypatch.setattr(ingest, "vector_store", store)
    monkeypatch.setattr(ingest, "bm25_manager", bm25)
    monkeypatch.setattr(ingest, "encode_texts", _embed)
    monkeypatch.setattr(ingest, "save_persisted_index", lambda: True)
    monkeypatch.setattr(ingest, "load_document", _load)
    monkeypatch.setattr(ingest, "split_blocks", lambda blocks: [
        Chunk(text=b.text, source=b.source, page=b.page, chunk_index=i, ocr_used=b.ocr_used)
        for i, b in enumerate(blocks)
    ])
    return store, bm25, clears


def test_sync_documents_accumulates_and_keeps_previous_document(kb, tmp_path):
    store, bm25, clears = kb
    (tmp_path / "a.pdf").write_bytes(b"a")

    first = ingest.sync_documents(str(tmp_path))
    assert first["added"] == ["a.pdf"]
    assert first["total_chunks"] == 1

    (tmp_path / "b.pdf").write_bytes(b"b")
    second = ingest.sync_documents(str(tmp_path))

    assert second["added"] == ["b.pdf"]
    assert second["skipped"] == ["a.pdf"]
    assert second["total_chunks"] == 2
    assert clears["count"] == 0                      # 增量入库绝不清空已有索引
    assert ("clear",) not in bm25.calls

    # 第一个文档仍可被检索（未被新文档覆盖）
    docs, _ids, metas, _d = store.search_with_scores(
        np.array([[1.0, 0.0, 0.0]], dtype="float32"), k=2)
    assert "alpha" in docs[0]
    assert metas[0]["source"] == "a.pdf"

    # BM25 用累计语料重建（含历史文档）
    assert bm25.calls[-1] == ("build", 2, 2)


def test_sync_documents_is_idempotent(kb, tmp_path):
    store, _bm25, clears = kb
    (tmp_path / "a.pdf").write_bytes(b"a")

    ingest.sync_documents(str(tmp_path))
    again = ingest.sync_documents(str(tmp_path))

    assert again["added"] == []
    assert again["skipped"] == ["a.pdf"]
    assert again["total_chunks"] == 1
    assert store.total_chunks == 1
    assert "已是最新" in again["message"]


def test_sync_documents_missing_dir_is_noop_without_model_load(monkeypatch, tmp_path):
    called = {"embed": False}

    def _spy(texts, show_progress=False):
        called["embed"] = True
        return np.zeros((len(texts), 3), dtype="float32")

    monkeypatch.setattr(ingest, "encode_texts", _spy)
    result = ingest.sync_documents(str(tmp_path / "not-exist"))

    assert result["status"] == "success"
    assert result["total_files"] == 0
    assert called["embed"] is False


def test_scan_documents_filters_by_whitelist(tmp_path):
    (tmp_path / "keep.pdf").write_bytes(b"x")
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    (tmp_path / "skip.exe").write_bytes(b"x")

    names = [os.path.basename(p) for p in ingest.scan_documents(str(tmp_path))]

    assert names == ["keep.pdf", "keep.txt"]


def test_kb_refresh_endpoint_returns_summary(monkeypatch):
    monkeypatch.setattr(api_router, "sync_documents", lambda *a, **k: {
        "status": "success", "added": ["x.pdf"], "skipped": [], "files": [],
        "total_files": 1, "total_chunks": 3, "persisted": True, "message": "ok",
    })
    resp = TestClient(api_router.app).post("/api/kb/refresh")

    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == ["x.pdf"]
    assert body["total_chunks"] == 3


class _FakeRequest:
    async def is_disconnected(self):
        return False


def test_stream_request_defaults_model_choice_to_config(monkeypatch):
    api_router.session_store.clear()
    seen = {}

    def fake_stream(question, session_id=None, history=None, enable_web_search=False,
                    model_choice=None, **kwargs):
        seen["model_choice"] = model_choice
        yield {"event": "delta", "data": {"text": "hi"}}
        yield {"event": "done", "data": {"latency_ms": 1}}

    monkeypatch.setattr(api_router, "stream_answer_events", fake_stream)
    payload = api_router.AskStreamRequest(question="q", session_id="kb-sess")

    frames = asyncio.run(_collect(api_router._sse_event_stream(_FakeRequest(), payload)))

    assert frames and frames[-1].startswith("event: done\n")
    assert seen["model_choice"] == api_router.DEFAULT_MODEL_CHOICE


async def _collect(agen):
    return [frame async for frame in agen]


def test_page_hides_engineering_details():
    with open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8") as handle:
        html = handle.read()
    with open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8") as handle:
        js = handle.read()

    # 用户页面不得再暴露上传入口、模型选择、索引/分数面板或调试接口。
    for token in ["file-input", "upload-btn", "force-ocr", "model-choice", "index-panel",
                  "index-badge", "retrieval-list", "retrieval-meta", "doc-list",
                  "/api/upload", "/api/status", "/api/documents"]:
        assert token not in html, token
        assert token not in js, token

    # 核心聊天能力仍在，且知识库同步走新的目录接口。
    for token in ["chat-log", "question", "ask-form", "ask-btn"]:
        assert token in html, token
    assert "/api/ask/stream" in js
    assert "/api/kb/refresh" in js
