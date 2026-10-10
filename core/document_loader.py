"""
文档加载器 —— 多格式文档文本提取（P2.2 按页 Block；P2.5 扫描件判定与 OCR）

学习要点：
- 了解不同文档格式（PDF、Word、Excel、PPT）的解析方式
- 理解 RAG 第一步：将非结构化文档转换为纯文本
- PDF 按「真实页码」逐页返回结构化 Block（WBS v1.0 P2.2 / 11.1），供扫描件判定
  （1.1.3「有效字符数 / 真实总页数」）使用
- 扫描件自动判定与 OCR 补足（WBS v1.0 P2.5）：判定走 §1.1.3 契约（每页平均字符数），
  文本型文档中的个别低文本页按页面级补足
"""

import os
import logging
from dataclasses import dataclass

import config


# P2.5 页面级阈值：有效字符数低于该值的页视为「低文本页」，在文本型文档中尝试 OCR 补足。
# 依据 WBS v1.0 P2.5「文本型文档中个别低文本页（<10 字符）走 OCR 补足」。
LOW_TEXT_PAGE_CHARS = 10


@dataclass
class Block:
    """按页提取的文档块（WBS v1.0 P2.2 / 11.1）。

    - ``text``：本页提取的纯文本（无文本层时为 ""）；
    - ``page``：PDF 页码，从 1 开始；非 PDF 统一为 0；
    - ``source``：文件名；
    - ``ocr_used``：本页是否经由 OCR 得到。P2.2 尚未接入 OCR，恒为 ``False``。
    """

    text: str
    page: int
    source: str
    ocr_used: bool = False


def load_document(filepath, force_ocr=False):
    """按页加载文档，返回 ``list[Block]``（P2.2，OCR 触发见 P2.5）。

    - PDF：用 ``pdfminer.high_level.extract_pages`` 逐页抽取，页数取 PDF 真实页数
      （无文本层的扫描页同样占一页，其 ``text`` 为空串），页码从 1 开始；
      随后按 WBS P2.5 契约判定是否需要 OCR（``force_ocr`` 强制跳过判定）；
    - 非 PDF（docx/xlsx/pptx/md/txt 等）：整体作为单一 Block 返回，``page=0``。

    文本顺序由 pdfminer 的页序与页内布局顺序决定，确定且可复现。
    """
    file_ext = os.path.splitext(filepath)[1].lower()
    source = os.path.basename(filepath)

    if file_ext == '.pdf':
        blocks = _load_pdf_blocks(filepath, source)
        return _apply_ocr_if_needed(filepath, blocks, force_ocr=force_ocr)

    return [Block(text=_extract_flat_text(filepath), page=0, source=source, ocr_used=False)]


def count_effective_chars(text):
    """统计「有效字符数」：非空白字符个数（WBS §1.1.3 判定分子）。"""
    return sum(1 for ch in text if not ch.isspace())


def is_scanned_document(total_effective_chars, real_page_count, threshold=None):
    """按 §1.1.3 契约判定扫描件：``有效字符数 / 真实总页数 < SCAN_TEXT_THRESHOLD``。

    注意使用**每页平均字符数**而非总字符数（否则「200 页仅 1 页扫描」会被误判为文本型）。
    ``real_page_count`` 必须来自 PDF 真实页数。
    """
    if real_page_count <= 0:
        return False
    limit = config.SCAN_TEXT_THRESHOLD if threshold is None else threshold
    average = total_effective_chars / real_page_count
    return average < limit


def _apply_ocr_if_needed(filepath, blocks, force_ocr=False):
    """按 P2.5 契约决定是否 OCR，并把 OCR 文本写回对应 Block。

    - ``OCR_ENABLED=false`` 时直接返回文本层结果；
    - ``force_ocr`` 或整篇判定为扫描件 → 逐页 OCR；
    - 文本型文档中个别低文本页（< ``LOW_TEXT_PAGE_CHARS``）→ 仅对这些页 OCR 补足并告警；
    - OCR 组件不可用或单页失败时不中断，保留文本层内容。
    """
    if not getattr(config, "OCR_ENABLED", True):
        logging.info("OCR_ENABLED=false，跳过扫描件判定，仅使用文本层")
        return blocks

    page_count = len(blocks)
    if page_count == 0:
        return blocks

    total_chars = sum(count_effective_chars(block.text) for block in blocks)
    scanned = is_scanned_document(total_chars, page_count)
    logging.info(
        "扫描件判定：有效字符数=%d，真实页数=%d，阈值=%d → %s",
        total_chars, page_count, config.SCAN_TEXT_THRESHOLD, "扫描件" if scanned else "文本型",
    )

    if force_ocr:
        targets = [block.page for block in blocks]
        logging.info("force_ocr=True，强制对全部 %d 页执行 OCR", page_count)
    elif scanned:
        targets = [block.page for block in blocks]
    else:
        targets = [
            block.page for block in blocks if count_effective_chars(block.text) < LOW_TEXT_PAGE_CHARS
        ]
        if not targets:
            return blocks
        logging.warning(
            "文本型文档中检测到 %d 个低文本页（有效字符 < %d），将按页面级 OCR 补足：页码 %s",
            len(targets), LOW_TEXT_PAGE_CHARS, targets,
        )

    return _ocr_blocks(filepath, blocks, targets)


def _ocr_blocks(filepath, blocks, target_pages):
    """对指定页执行 OCR，并把结果写回 Block（失败时保留原文本层内容）。"""
    try:
        from core import ocr as ocr_module
    except Exception as exc:  # OCR 组件缺失不应让整篇加载失败
        logging.error("OCR 组件不可用，保留文本层结果：%s", exc)
        return blocks

    target_pages = ocr_module.truncate_pages(target_pages)
    try:
        rendered = {page.page: page for page in ocr_module.render_pdf_pages(filepath)}
    except Exception as exc:
        logging.error("PDF 渲染失败，保留文本层结果：%s", exc)
        return blocks

    by_page = {block.page: block for block in blocks}
    for page_number in target_pages:
        rendered_page = rendered.get(page_number)
        if rendered_page is None or rendered_page.status != "ok" or rendered_page.image is None:
            logging.warning("第 %d 页渲染失败（render_failed），保留原文本层内容", page_number)
            continue
        try:
            result = ocr_module.ocr_image(rendered_page.image)
        except Exception as exc:
            logging.error("第 %d 页 OCR 失败，保留原文本层内容：%s", page_number, exc)
            continue
        block = by_page.get(page_number)
        if block is None:
            continue
        if result.text:
            block.text = result.text
        block.ocr_used = True
    return blocks


def _load_pdf_blocks(filepath, source):
    """按 PDF 真实页序逐页抽取文本，返回 Block 列表（页码从 1 开始）。"""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    blocks = []
    for page_number, page_layout in enumerate(extract_pages(filepath), start=1):
        parts = [
            element.get_text()
            for element in page_layout
            if isinstance(element, LTTextContainer)
        ]
        blocks.append(
            Block(text="".join(parts), page=page_number, source=source, ocr_used=False)
        )
    return blocks


def extract_text(filepath):
    """兼容包装：返回整份文档的纯文本（各 Block 以换行拼接）。

    保留原 ``extract_text(filepath) -> str`` 签名与行为，既有调用方
    （``rag_demo.py``）与 ``tests/test_document_loader.py`` 无需改动。
    """
    return "\n".join(block.text for block in load_document(filepath))


def _extract_flat_text(filepath):
    """非 PDF 格式的扁平文本提取（原 ``extract_text`` 主体逻辑）。"""
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    elif file_ext == '.docx':
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            logging.error("处理Word文档需要安装python-docx库")
            return ""

    elif file_ext in ['.xlsx', '.xls']:
        try:
            import pandas as pd
            text = ""
            xl = pd.ExcelFile(filepath)
            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                text += f"工作表: {sheet_name}\n"
                text += df.to_string(index=False) + "\n\n"
            return text
        except ImportError:
            logging.error("处理Excel文件需要安装pandas库")
            return ""

    elif file_ext == '.pptx':
        try:
            from pptx import Presentation
            prs = Presentation(filepath)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        except ImportError:
            logging.error("处理PPT文件需要安装python-pptx库")
            return ""

    else:
        logging.warning(f"不支持的文件格式: {file_ext}")
        return ""
