"""OCR 引擎接入、PDF 页渲染与阅读顺序整理（P2.3 / P2.4 / P2.6）。

WBS v1.0 依据：
- P2.3：引擎选型 RapidOCR（onnxruntime）——纯 pip 安装、模型随 wheel 内置、完全离线
        可跑；引擎实例懒加载并缓存；首次与后续调用耗时分别记录；单文档 OCR 页数超过
        ``OCR_MAX_PAGES`` 时截断并告警。
- P2.4：用 PyMuPDF（fitz）把 PDF 页渲染为位图，DPI 走 ``OCR_RENDER_DPI``；单页渲染失败
        记为 ``render_failed`` 并继续其余页，不中断整篇。
- P2.6：利用文本框坐标按 y 排序近似恢复阅读顺序，同一行内按 x 排序；只做轻量近似，
        不做复杂版面还原（WBS §14 明确不实现多栏重排）。
"""

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# P2.6 段落合并阈值：相邻两行「行框上下间距」（下一行顶部 - 上一行底部）< 该比例 × 参考
# 行高时，视为同一段落。取值依据：常规排版的行间留白远小于行高；段落之间的留白通常
# ≥ 1 倍行高，故取 0.6 倍行高可在「行内合并」与「段落分隔」之间留出余量。
PARAGRAPH_Y_RATIO = 0.6


@dataclass
class OcrLine:
    """单条 OCR 文本及其几何信息。``box`` = (x_min, y_min, x_max, y_max)。"""

    text: str
    box: Tuple[float, float, float, float]
    score: float
    y_center: float
    height: float


@dataclass
class OcrResult:
    """一张图片的 OCR 结果：阅读顺序整理后的文本 + 逐行明细。"""

    text: str
    lines: List[OcrLine] = field(default_factory=list)
    engine: str = "rapidocr"


@dataclass
class RenderedPage:
    """单页 PDF 渲染结果。``status`` 取值 ``"ok"`` / ``"render_failed"``。"""

    page: int
    status: str
    image: Optional[np.ndarray] = None
    error: str = ""


# ── OCR 引擎：懒加载 + 缓存（P2.3）────────────────────────────────────
_ENGINE = None
_FIRST_CALL_DONE = False


def _get_engine():
    """懒加载并缓存 RapidOCR 引擎实例，避免每次调用重新初始化。"""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    engine_name = getattr(config, "OCR_ENGINE", "rapidocr")
    if engine_name != "rapidocr":
        raise ValueError(f"不支持的 OCR_ENGINE={engine_name!r}（WBS 仅规定 rapidocr）")
    init_started = time.perf_counter()
    from rapidocr_onnxruntime import RapidOCR

    _ENGINE = RapidOCR()
    logger.info("RapidOCR 引擎初始化完成，耗时 %.3fs", time.perf_counter() - init_started)
    return _ENGINE


def reset_engine():
    """清空缓存的引擎实例（供测试或显式重载使用）。"""
    global _ENGINE, _FIRST_CALL_DONE
    _ENGINE = None
    _FIRST_CALL_DONE = False


def _box_to_rect(box):
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return (min(xs), min(ys), max(xs), max(ys))


def _parse_raw(raw):
    """把 RapidOCR 原始输出 ``[[box, text, score], ...]`` 解析为 ``OcrLine`` 列表。"""
    lines = []
    if not raw:
        return lines
    for item in raw:
        box, text, score = item[0], item[1], item[2]
        rect = _box_to_rect(box)
        lines.append(
            OcrLine(
                text=str(text),
                box=rect,
                score=float(score),
                y_center=(rect[1] + rect[3]) / 2.0,
                height=rect[3] - rect[1],
            )
        )
    return lines


def ocr_image(image) -> OcrResult:
    """对单张图片执行 OCR，返回按阅读顺序整理后的 ``OcrResult``（P2.3 / P2.6）。"""
    global _FIRST_CALL_DONE
    engine = _get_engine()
    started = time.perf_counter()
    raw, _elapse = engine(image)
    elapsed = time.perf_counter() - started
    if not _FIRST_CALL_DONE:
        _FIRST_CALL_DONE = True
        logger.info("OCR 首次调用耗时 %.3fs", elapsed)
    else:
        logger.info("OCR 后续调用耗时 %.3fs", elapsed)

    ordered = order_lines(_parse_raw(raw))
    return OcrResult(
        text=lines_to_text(ordered),
        lines=ordered,
        engine=getattr(config, "OCR_ENGINE", "rapidocr"),
    )


# ── 阅读顺序整理（P2.6）─────────────────────────────────────────────
def _reference_height(lines):
    heights = [ln.height for ln in lines if ln.height > 0]
    return statistics.median(heights) if heights else 1.0


def _x_overlap(a, b):
    return min(a.box[2], b.box[2]) > max(a.box[0], b.box[0])


def _join(left, right):
    """拼接两段 OCR 文本。

    若两侧边界均为 ASCII 字母/数字，则补一个空格（英文等以空白分词的语言需要）；
    否则直接拼接（中文无需空格）。仅影响同一段落内的换行拼接，不改变行序。
    """
    if (
        left
        and right
        and left[-1].isascii()
        and left[-1].isalnum()
        and right[0].isascii()
        and right[0].isalnum()
    ):
        return left + " " + right
    return left + right


def order_lines(lines, row_ratio=PARAGRAPH_Y_RATIO):
    """按 y 坐标升序排列，同一行（y 相近）内按 x 升序排列（P2.6）。

    "同一行" 的判定阈值 = ``row_ratio`` × 参考行高，避免逐像素抖动导致行序不稳定。
    """
    if not lines:
        return []
    tolerance = row_ratio * _reference_height(lines)
    rows = []
    for line in sorted(lines, key=lambda ln: ln.y_center):
        if rows and abs(line.y_center - rows[-1][0]) <= tolerance:
            rows[-1][1].append(line)
        else:
            rows.append((line.y_center, [line]))
    ordered = []
    for _y, row_lines in rows:
        ordered.extend(sorted(row_lines, key=lambda ln: ln.box[0]))
    return ordered


def lines_to_text(ordered_lines, y_ratio=PARAGRAPH_Y_RATIO):
    """把已排序的行合并为文本（P2.6）。

    相邻两行的「y 间距」（下一行顶部 - 上一行底部）小于阈值且水平方向重叠时，视为同一
    段落（直接拼接）；否则另起一行。水平不重叠（并排）的行不会被拼接，从而避免把不同栏
    的内容错拼到同一行。
    """
    if not ordered_lines:
        return ""
    tolerance = y_ratio * _reference_height(ordered_lines)
    paragraphs = []
    previous = None
    for line in ordered_lines:
        if previous is None:
            paragraphs.append(line.text)
        else:
            vertical_gap = line.box[1] - previous.box[3]
            same_paragraph = vertical_gap < tolerance and _x_overlap(previous, line)
            if same_paragraph:
                paragraphs[-1] = _join(paragraphs[-1], line.text)
            else:
                paragraphs.append(line.text)
        previous = line
    return "\n".join(paragraphs)


# ── PDF 页渲染（P2.4）───────────────────────────────────────────────
def _pixmap_to_bgr(pix):
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        return np.ascontiguousarray(array[:, :, ::-1])  # RGB → BGR（RapidOCR/cv2 约定）
    if pix.n == 4:
        return np.ascontiguousarray(array[:, :, 2::-1])
    if pix.n == 1:
        return np.ascontiguousarray(np.repeat(array, 3, axis=2))
    return np.ascontiguousarray(array)


def render_pdf_pages(pdf_path, dpi=None):
    """逐页把 PDF 渲染为 BGR 位图（P2.4），返回 ``list[RenderedPage]``。

    - DPI 取参数或 ``config.OCR_RENDER_DPI``（默认 200）；
    - 单页渲染失败（加密/损坏页等）→ 该页 ``status="render_failed"`` 并继续其余页；
    - PDF 整体无法打开 → 记录错误并返回空列表，不抛异常。
    """
    import fitz  # PyMuPDF

    render_dpi = dpi or getattr(config, "OCR_RENDER_DPI", 200)
    zoom = render_dpi / 72.0
    results = []
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # 加密/损坏导致整份无法打开
        logger.error("PDF 打开失败，无法渲染任何页（render_failed）：%s", exc)
        return results
    try:
        for index in range(document.page_count):
            page_number = index + 1
            try:
                page = document.load_page(index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                results.append(
                    RenderedPage(page=page_number, status="ok", image=_pixmap_to_bgr(pixmap))
                )
            except Exception as exc:
                logger.warning("PDF 第 %d 页渲染失败（render_failed），跳过该页：%s", page_number, exc)
                results.append(
                    RenderedPage(page=page_number, status="render_failed", error=str(exc))
                )
    finally:
        document.close()
    return results


# ── 多页 OCR 与页数截断（P2.3）───────────────────────────────────────
def resolve_max_pages(max_pages=None):
    """返回生效的 OCR 页数上限（参数优先，其次 ``config.OCR_MAX_PAGES``）。"""
    if max_pages is not None:
        return max_pages
    return getattr(config, "OCR_MAX_PAGES", 50)


def truncate_pages(items, max_pages=None, what="待 OCR 页"):
    """超过 ``OCR_MAX_PAGES`` 时截断并告警（P2.3），不静默处理。"""
    limit = resolve_max_pages(max_pages)
    if limit and len(items) > limit:
        logger.warning(
            "%s数量 %d 超过 OCR_MAX_PAGES=%d，截断为前 %d 页（P2.3）",
            what, len(items), limit, limit,
        )
        return items[:limit]
    return items


def ocr_pages(images, max_pages=None):
    """对多页位图依次 OCR，超过页数上限时截断并告警（P2.3）。"""
    return [ocr_image(image) for image in truncate_pages(images, max_pages=max_pages)]
