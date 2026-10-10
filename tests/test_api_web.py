"""P4.1 / P4.2 / P4.6：Web 前端托管与 REST 契约测试（静态资源、上传、文档列表、状态）。

不加载模型、不联网；上传入口已停用（P6.1），用真实 VectorStore 验证其不会清空已有索引。
"""

import os

from fastapi.testclient import TestClient

import api_router

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _client():
    return TestClient(api_router.app)


def test_index_page_served():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "本地 RAG" in resp.text
    assert "app.js" in resp.text and "app.css" in resp.text


def test_static_assets_served():
    client = _client()
    css = client.get("/static/app.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def test_index_html_uses_no_frontend_framework():
    with open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8") as handle:
        html = handle.read().lower()
    for token in ["react", "vue", "angular", "svelte"]:
        assert token not in html


def test_gradio_entry_and_dependency_removed():
    """P4.1 验收：Gradio 入口与依赖已移除。"""
    assert not os.path.exists(os.path.join(ROOT, "rag_demo.py"))
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
        # 只看真实依赖行，忽略注释（注释里说明「已移除 gradio」是允许的）。
        requirements = [ln.strip() for ln in handle
                        if ln.strip() and not ln.strip().startswith("#")]
    assert not any("gradio" in line.lower() for line in requirements)


def test_upload_endpoint_is_disabled_with_clear_message():
    """P6.1：前端上传入口已移除，POST /api/upload 返回明确错误并指向目录同步。"""
    client = _client()
    resp = client.post("/api/upload", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert resp.status_code == 410
    detail = resp.json()["detail"]
    assert "知识库目录同步模式" in detail
    assert "data/documents" in detail

    # 即使不提供文件，也不再是 400/200，而是统一的「已停用」错误。
    assert client.post("/api/upload", data={}).status_code == 410


def test_upload_does_not_clear_or_rebuild_existing_index(monkeypatch):
    """P6.1：上传入口关闭后，已有索引不会被清空，也不会再触发 ingest_files(reset=True)。"""
    import numpy as np
    import core.ingest as ingest
    from core.vector_store import vector_store

    def _forbidden(*args, **kwargs):
        raise AssertionError("已停用的 /api/upload 不得再调用 ingest_files")

    monkeypatch.setattr(ingest, "ingest_files", _forbidden)

    vector_store.clear()
    vector_store.build_index(
        ["keep-me"], ["c1"], [{"source": "keep.pdf", "page": 1}],
        np.zeros((1, 4), dtype="float32"),
    )
    try:
        resp = _client().post(
            "/api/upload", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
        assert resp.status_code == 410
        assert vector_store.total_chunks == 1            # 旧索引仍在
        assert "c1" in vector_store.contents_map
    finally:
        vector_store.clear()


def test_documents_endpoint_returns_list(monkeypatch):
    monkeypatch.setattr(api_router, "list_documents", lambda: [
        {"source": "a.pdf", "doc_id": "d1", "chunks": 2, "ocr_used": True, "page_count": 1},
    ])
    body = _client().get("/api/documents").json()
    assert body["documents"][0]["source"] == "a.pdf"
    assert "total_chunks" in body and "index_ready" in body


def test_status_exposes_index_state_and_models():
    body = _client().get("/api/status").json()
    for key in ["index_status", "index_message", "index_dir", "total_chunks",
                "indexed_documents", "embedding_model", "index_built_at", "models",
                "model_choices", "default_model", "version"]:
        assert key in body, key
    assert "嵌入模型" in body["models"]
    assert body["model_choices"] and {"value", "label"} <= set(body["model_choices"][0])
