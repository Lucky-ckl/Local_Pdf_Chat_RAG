"""P2.3-P2.6 单测：OCR 引擎契约、页数截断、阅读顺序、扫描件判定、PDF 渲染。

按 WBS v1.0 P2.8：OCR 引擎本身不做端到端文本断言（结果受字体影响不稳定），只验证调用
契约与错误处理；扫描件判定（文本型/扫描型/混合型/空文档）、渲染失败路径、排序逻辑均覆盖。
"""

import logging

import numpy as np
import pytest

from core import document_loader as dl
from core import ocr


@pytest.fixture(autouse=True)
def _reset_engine():
    ocr.reset_engine()
    yield
    ocr.reset_engine()


# ── P2.5 扫描件判定：文本型 / 扫描型 / 混合型 / 空文档 ──────────────────
def test_scan_detection_text_document_is_not_scanned():
    # 10 页共 3000 有效字符 → 每页 300 ≥ 阈值 50 → 文本型
    assert dl.is_scanned_document(3000, 10) is False


def test_scan_detection_scanned_document():
    # 10 页共 10 有效字符 → 每页 1 < 阈值 50 → 扫描件
    assert dl.is_scanned_document(10, 10) is True


def test_scan_detection_mixed_200_pages_one_scan():
    # 200 页仅 1 页为扫描图：按总量判定不得误判为扫描件（WBS §1.1.3 必测项）
    total = 199 * 200  # 199 页各 200 字符，第 200 页为 0 字符
    assert dl.is_scanned_document(total, 200) is False
    # 该扫描页命中页面级低文本判定（< LOW_TEXT_PAGE_CHARS）
    assert dl.count_effective_chars("") < dl.LOW_TEXT_PAGE_CHARS


def test_scan_detection_empty_document_is_not_scanned():
    assert dl.is_scanned_document(0, 0) is False


def test_scan_detection_uses_per_page_average():
    # 直接验证「每页平均」而非「总量」：总量相同，页数不同 → 判定不同
    assert dl.is_scanned_document(4000, 200) is True   # 平均 20 < 50
    assert dl.is_scanned_document(4000, 40) is False   # 平均 100 ≥ 50


# ── P2.3 引擎懒加载与缓存 ────────────────────────────────────────────
def test_engine_is_lazy_loaded_and_cached(monkeypatch):
    import rapidocr_onnxruntime

    created = {"count": 0}

    class _FakeRapidOCR:
        def __init__(self):
            created["count"] += 1

        def __call__(self, image):
            return (None, [0.0, 0.0, 0.0])

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", _FakeRapidOCR)

    assert created["count"] == 0  # 懒加载：调用前不初始化
    ocr.ocr_image(object())
    ocr.ocr_image(object())
    assert created["count"] == 1  # 缓存复用：仅初始化一次


def test_engine_prefers_cuda_provider_when_available(monkeypatch, caplog):
    """CUDA 可用时：以 use_cuda=True 构造引擎，并记录 OCR Provider: CUDA。"""
    import rapidocr_onnxruntime

    seen = {}

    class _Session:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class _Backend:
        def __init__(self):
            self.session = _Session()

    class _Infer:
        def __init__(self):
            self.infer = _Backend()

    class _FakeRapidOCR:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.text_det = _Infer()
            self.text_rec = _Infer()
            self.text_cls = _Infer()

        def __call__(self, image):
            return (None, [0.0, 0.0, 0.0])

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", _FakeRapidOCR)
    monkeypatch.setattr(ocr, "_cuda_provider_available", lambda: True)

    with caplog.at_level(logging.INFO, logger="core.ocr"):
        ocr.ocr_image(object())

    assert seen == {"det_use_cuda": True, "rec_use_cuda": True, "cls_use_cuda": True}
    assert ocr.get_engine_provider() == "CUDA"
    assert any("OCR Provider: CUDA" in r.getMessage() for r in caplog.records)


def test_engine_falls_back_to_cpu_when_cuda_init_fails(monkeypatch, caplog):
    """CUDA 可用但初始化抛错时：自动回退 CPU，并记录 OCR Provider: CPU。"""
    import rapidocr_onnxruntime

    calls = []

    class _FakeRapidOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs:
                raise RuntimeError("CUDA runtime not found")

        def __call__(self, image):
            return (None, [0.0, 0.0, 0.0])

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", _FakeRapidOCR)
    monkeypatch.setattr(ocr, "_cuda_provider_available", lambda: True)

    with caplog.at_level(logging.INFO, logger="core.ocr"):
        ocr.ocr_image(object())

    assert calls == [{"det_use_cuda": True, "rec_use_cuda": True, "cls_use_cuda": True}, {}]
    assert ocr.get_engine_provider() == "CPU"
    assert any("OCR Provider: CPU" in r.getMessage() for r in caplog.records)


def test_engine_uses_cpu_when_cuda_provider_missing(monkeypatch, caplog):
    """无 CUDA Provider 时：直接走 CPU，并记录 OCR Provider: CPU。"""
    import rapidocr_onnxruntime

    class _FakeRapidOCR:
        def __call__(self, image):
            return (None, [0.0, 0.0, 0.0])

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", _FakeRapidOCR)
    monkeypatch.setattr(ocr, "_cuda_provider_available", lambda: False)

    with caplog.at_level(logging.INFO, logger="core.ocr"):
        ocr.ocr_image(object())

    assert ocr.get_engine_provider() == "CPU"
    assert any("OCR Provider: CPU" in r.getMessage() for r in caplog.records)


def test_engine_reports_cpu_when_cuda_session_falls_back(monkeypatch, caplog):
    """CUDA provider 可用但会话实际回退 CPU：不得误报 CUDA，须记 CPU 并告警。"""
    import rapidocr_onnxruntime

    class _Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class _Backend:
        def __init__(self):
            self.session = _Session()

    class _Infer:
        def __init__(self):
            self.infer = _Backend()

    class _FakeRapidOCR:
        def __init__(self, **kwargs):
            self.text_det = _Infer()
            self.text_rec = _Infer()
            self.text_cls = _Infer()

        def __call__(self, image):
            return (None, [0.0, 0.0, 0.0])

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", _FakeRapidOCR)
    monkeypatch.setattr(ocr, "_cuda_provider_available", lambda: True)

    with caplog.at_level(logging.INFO, logger="core.ocr"):
        ocr.ocr_image(object())

    assert ocr.get_engine_provider() == "CPU"
    assert any("OCR Provider: CPU" in r.getMessage() for r in caplog.records)
    assert any("fallback to CPU" in r.getMessage() for r in caplog.records)


def test_preload_failure_does_not_block_engine_init(monkeypatch, caplog):
    """preload_dlls 抛错时不得阻断引擎初始化（无 GPU 或环境损坏仍可运行）。"""
    import onnxruntime
    import rapidocr_onnxruntime

    def _boom():
        raise RuntimeError("preload boom")

    class _FakeRapidOCR:
        def __call__(self, image):
            return (None, [0.0, 0.0, 0.0])

    monkeypatch.setattr(onnxruntime, "preload_dlls", _boom, raising=False)
    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", _FakeRapidOCR)
    monkeypatch.setattr(ocr, "_cuda_provider_available", lambda: False)

    with caplog.at_level(logging.WARNING, logger="core.ocr"):
        ocr.ocr_image(object())

    assert ocr.get_engine_provider() == "CPU"
    assert any("preload" in r.getMessage() for r in caplog.records)


def test_ocr_image_orders_lines_and_reports_engine(monkeypatch):
    import rapidocr_onnxruntime

    raw = [
        [[[100, 40], [180, 40], [180, 60], [100, 60]], "SECOND", 0.9],
        [[[10, 40], [90, 40], [90, 60], [10, 60]], "FIRST", 0.9],
    ]

    class _FakeRapidOCR:
        def __call__(self, image):
            return (raw, [0.0, 0.0, 0.0])

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", _FakeRapidOCR)

    result = ocr.ocr_image(object())

    assert isinstance(result, ocr.OcrResult)
    assert [line.text for line in result.lines] == ["FIRST", "SECOND"]
    assert result.text == "FIRST\nSECOND"
    assert result.engine == "rapidocr"


# ── P2.3 页数截断与告警 ─────────────────────────────────────────────
def test_truncate_pages_warns_and_limits(caplog):
    with caplog.at_level(logging.WARNING):
        out = ocr.truncate_pages(list(range(5)), max_pages=3)

    assert out == [0, 1, 2]
    assert any("OCR_MAX_PAGES" in record.getMessage() for record in caplog.records)


def test_ocr_pages_truncates_before_engine_call(monkeypatch):
    seen = []
    monkeypatch.setattr(
        ocr, "ocr_image", lambda image: seen.append(image) or ocr.OcrResult(text="x")
    )

    results = ocr.ocr_pages([1, 2, 3, 4, 5], max_pages=2)

    assert len(results) == 2
    assert seen == [1, 2]


# ── P2.6 阅读顺序 ───────────────────────────────────────────────────
def test_order_lines_sorts_by_y_then_x():
    lines = [
        ocr.OcrLine("B", (200, 100, 300, 120), 0.9, 110.0, 20.0),
        ocr.OcrLine("A", (10, 100, 100, 120), 0.9, 110.0, 20.0),
        ocr.OcrLine("C", (10, 130, 100, 150), 0.9, 140.0, 20.0),
    ]

    assert [line.text for line in ocr.order_lines(lines)] == ["A", "B", "C"]


def test_lines_to_text_merges_vertically_stacked_lines_only():
    # 同栏相邻两行（y 间距 5 < 0.6×20）→ 合并为同一段落
    stacked = [
        ocr.OcrLine("第一", (10, 100, 100, 120), 0.9, 110.0, 20.0),
        ocr.OcrLine("段。", (10, 125, 100, 145), 0.9, 135.0, 20.0),
    ]
    assert ocr.lines_to_text(stacked) == "第一段。"

    # 并排（水平不重叠）的行不得拼接，避免跨栏错拼
    side_by_side = [
        ocr.OcrLine("左栏", (10, 100, 90, 120), 0.9, 110.0, 20.0),
        ocr.OcrLine("右栏", (200, 100, 280, 120), 0.9, 110.0, 20.0),
    ]
    assert ocr.lines_to_text(side_by_side) == "左栏\n右栏"


# ── P2.4 PDF 渲染 ──────────────────────────────────────────────────
def test_render_pdf_pages_renders_real_page(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "one_page.pdf"
    document = fitz.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 50), "HELLO")
    document.save(str(pdf_path))
    document.close()

    pages = ocr.render_pdf_pages(str(pdf_path), dpi=150)

    assert len(pages) == 1
    assert pages[0].status == "ok"
    assert isinstance(pages[0].image, np.ndarray)
    assert pages[0].image.ndim == 3


def test_render_pdf_pages_marks_failed_page_and_continues(monkeypatch):
    fitz = pytest.importorskip("fitz")

    class _FakePixmap:
        height = 4
        width = 4
        n = 3
        samples = bytes(4 * 4 * 3)

    class _FakePage:
        def __init__(self, broken):
            self._broken = broken

        def get_pixmap(self, matrix=None, alpha=False):
            if self._broken:
                raise RuntimeError("损坏页")
            return _FakePixmap()

    class _FakeDoc:
        page_count = 2

        def load_page(self, index):
            return _FakePage(broken=(index == 1))

        def close(self):
            pass

    monkeypatch.setattr(fitz, "open", lambda path: _FakeDoc())

    pages = ocr.render_pdf_pages("dummy.pdf")

    assert [page.status for page in pages] == ["ok", "render_failed"]
    assert pages[0].image is not None
    assert pages[1].image is None
