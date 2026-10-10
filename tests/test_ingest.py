"""P4.1 / P4.2：文档入库编排（core/ingest.py）单测。

全部通过 monkeypatch 隔离：不加载 embedding 模型、不构建真实索引、不联网。
"""

import pytest

import core.ingest as ingest
from core.document_loader import Block
from core.text_splitter import Chunk


class _FakeVectorStore:
    def __init__(self):
        self.metadatas_map = {}
        self.contents_map = {}
        self.id_order = []
        self.total_chunks = 0
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1
        self.metadatas_map, self.contents_map, self.id_order, self.total_chunks = {}, {}, [], 0

    def build_index(self, chunks, ids, metadatas, embeddings):
        self.contents_map = dict(zip(ids, chunks))
        self.metadatas_map = dict(zip(ids, metadatas))
        self.id_order = list(ids)
        self.total_chunks = len(ids)


class _FakeBM25:
    def __init__(self):
        self.built = None
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1

    def build_index(self, chunks, ids):
        self.built = (len(chunks), len(ids))


@pytest.fixture
def fake_env(monkeypatch):
    vector = _FakeVectorStore()
    bm25 = _FakeBM25()
    monkeypatch.setattr(ingest, "vector_store", vector)
    monkeypatch.setattr(ingest, "bm25_manager", bm25)
    monkeypatch.setattr(ingest, "encode_texts", lambda texts, show_progress=False: [[0.0] * 3 for _ in texts])
    monkeypatch.setattr(ingest, "save_persisted_index", lambda: True)
    return vector, bm25


def _one_chunk(blocks, source="s.pdf", page=1, ocr_used=False):
    return [Chunk(text=b.text, source=source, page=page, chunk_index=i, ocr_used=ocr_used)
            for i, b in enumerate(blocks)]


def test_ingest_success_builds_index_and_persists(fake_env, monkeypatch, tmp_path):
    vector, bm25 = fake_env
    monkeypatch.setattr(ingest, "load_document",
                        lambda path, force_ocr=False: [Block(text="x", page=1, source="s.pdf")])
    monkeypatch.setattr(ingest, "split_blocks", _one_chunk)

    a = tmp_path / "a.txt"; a.write_text("a", encoding="utf-8")
    b = tmp_path / "b.txt"; b.write_text("b", encoding="utf-8")

    result = ingest.ingest_files([str(a), str(b)])

    assert result["status"] == "success"
    assert result["persisted"] is True
    assert [f["status"] for f in result["files"]] == ["success", "success"]
    assert result["total_chunks"] == 2
    assert vector.total_chunks == 2
    assert bm25.built == (2, 2)
    assert vector.clear_count == 1 and bm25.clear_count == 1


def test_ingest_single_file_failure_continues(fake_env, monkeypatch, tmp_path):
    def load(path, force_ocr=False):
        if path.endswith("bad.txt"):
            raise ValueError("boom")
        return [Block(text="ok", page=0, source="ok.txt")]

    monkeypatch.setattr(ingest, "load_document", load)
    monkeypatch.setattr(ingest, "split_blocks", _one_chunk)
    good = tmp_path / "good.txt"; good.write_text("x", encoding="utf-8")
    bad = tmp_path / "bad.txt"; bad.write_text("x", encoding="utf-8")

    result = ingest.ingest_files([str(bad), str(good)])
    by_name = {f["filename"]: f for f in result["files"]}

    assert by_name["bad.txt"]["status"] == "error"
    assert by_name["good.txt"]["status"] == "success"
    assert result["status"] == "success"      # 至少一个成功
    assert result["total_chunks"] == 1


def test_ingest_rejects_unsupported_extension(fake_env, monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "load_document",
                        lambda path, force_ocr=False: pytest.fail("不应加载不支持的类型"))
    bad = tmp_path / "a.exe"; bad.write_text("x", encoding="utf-8")

    result = ingest.ingest_files([str(bad)])

    assert result["status"] == "error"
    assert "不支持的文件类型" in result["files"][0]["message"]
    assert result["total_chunks"] == 0


def test_ingest_reports_ocr_used(fake_env, monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "load_document",
                        lambda path, force_ocr=False: [Block(text="t", page=1, source="s.pdf", ocr_used=True)])
    monkeypatch.setattr(ingest, "split_blocks", lambda blocks: _one_chunk(blocks, ocr_used=True))
    pdf = tmp_path / "s.pdf"; pdf.write_text("x", encoding="utf-8")

    result = ingest.ingest_files([str(pdf)])

    assert result["files"][0]["ocr_used"] is True
    assert "已启用 OCR" in result["files"][0]["message"]


def test_ingest_passes_force_ocr(fake_env, monkeypatch, tmp_path):
    seen = {}

    def load(path, force_ocr=False):
        seen["force_ocr"] = force_ocr
        return [Block(text="t", page=1, source="s.pdf")]

    monkeypatch.setattr(ingest, "load_document", load)
    monkeypatch.setattr(ingest, "split_blocks", _one_chunk)
    pdf = tmp_path / "s.pdf"; pdf.write_text("x", encoding="utf-8")

    ingest.ingest_files([str(pdf)], force_ocr=True)

    assert seen["force_ocr"] is True


def test_ingest_empty_paths_returns_error():
    result = ingest.ingest_files([])
    assert result["status"] == "error"
    assert result["total_chunks"] == 0
    assert result["files"] == []


def test_list_documents_aggregates_real_metadata(monkeypatch):
    vector = _FakeVectorStore()
    vector.metadatas_map = {
        "c1": {"source": "a.pdf", "doc_id": "d1", "page": 1, "chunk_index": 0, "ocr_used": True},
        "c2": {"source": "a.pdf", "doc_id": "d1", "page": 3, "chunk_index": 1, "ocr_used": False},
        "c3": {"source": "b.txt", "doc_id": "d2", "page": 0, "chunk_index": 0, "ocr_used": False},
    }
    monkeypatch.setattr(ingest, "vector_store", vector)

    by_source = {d["source"]: d for d in ingest.list_documents()}

    assert by_source["a.pdf"]["chunks"] == 2
    assert by_source["a.pdf"]["ocr_used"] is True
    assert by_source["a.pdf"]["page_count"] == 2      # 页 1 与 3（page=0 不计入页数）
    assert by_source["b.txt"]["page_count"] == 0
