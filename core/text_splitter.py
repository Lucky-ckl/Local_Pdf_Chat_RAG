"""
文本分块器 —— 将长文本切分为检索友好的片段（P2.7 补充 chunk 页码元数据）

学习要点：
- chunk_size：每个片段的最大字符数。过大则检索粒度粗，过小则上下文缺失
- chunk_overlap：相邻片段的重叠字符数。避免关键信息被切断
- separators：按优先级尝试的分割符。中文文档应包含中文标点

P2.7：在既有 ``split_text`` 之上新增 ``split_blocks``，把 P2.2 的按页 ``Block`` 切分为
带 ``source / page / chunk_index / ocr_used`` 元数据的 ``Chunk``；跨页切分时 ``page`` 取
**起始页**（WBS v1.0 P2.7 / 11.2）。
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """带页码元数据的文本块（WBS v1.0 P2.7 / 11.2）。

    - ``text``：切分后的文本；
    - ``source``：来源文件名；
    - ``page``：所在页（跨页 chunk 取起始页；非 PDF 为 0）；
    - ``chunk_index``：文档内的切分序号（从 0 开始）；
    - ``ocr_used``：该 chunk 覆盖的任一页走过 OCR 即为 True。

    ``id`` 不在本结构内，沿用上游既有 chunk_id 生成方式（由调用方基于 chunk_index 生成）。
    """

    text: str
    source: str
    page: int
    chunk_index: int
    ocr_used: bool


def split_text(text, chunk_size=None, chunk_overlap=None):
    """
    将长文本切分为多个片段

    使用 RecursiveCharacterTextSplitter 递归切分：
    先尝试按段落分割，若片段仍过大则按句子分割，以此类推。

    Args:
        text: 待切分的长文本
        chunk_size: 每个片段的最大字符数（默认使用配置值 400）
        chunk_overlap: 相邻片段的重叠字符数（默认使用配置值 40）

    Returns:
        切分后的文本片段列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return text_splitter.split_text(text)


def split_blocks(blocks, chunk_size=None, chunk_overlap=None):
    """把按页 ``Block`` 切分并补齐页码元数据（P2.7），返回 ``list[Chunk]``。

    实现：先按 ``extract_text`` 相同的规则把各页文本用换行拼接成全文，并记录每页起止
    字符偏移；再用 ``split_text`` 递归切分；最后把每个 chunk 的起始偏移映射回页码。

    - 跨页切分时 ``page`` 取**起始页**；
    - ``ocr_used`` 取该 chunk 覆盖范围内是否任意一页走过 OCR；
    - ``chunk_index`` 为文档内从 0 开始的自增序号。
    """
    if not blocks:
        return []

    parts = []
    page_spans = []  # (start_offset, end_offset, page, source, ocr_used)
    offset = 0
    for index, block in enumerate(blocks):
        if index > 0:
            parts.append("\n")
            offset += 1
        start = offset
        parts.append(block.text)
        offset += len(block.text)
        page_spans.append((start, offset, block.page, block.source, block.ocr_used))
    full_text = "".join(parts)

    pieces = split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    cursor = 0
    for index, piece in enumerate(pieces):
        position = full_text.find(piece, cursor)
        if position < 0:
            position = cursor
        cursor = position
        start_page, source = _page_for_offset(position, page_spans)
        ocr_used = _any_ocr_in_range(position, position + len(piece), page_spans)
        chunks.append(
            Chunk(
                text=piece,
                source=source,
                page=start_page,
                chunk_index=index,
                ocr_used=ocr_used,
            )
        )
    return chunks


def _page_for_offset(position, page_spans):
    """返回包含字符偏移 ``position`` 的页（页码、来源文件名）；取起始页语义。"""
    for start, end, page, source, _ocr in page_spans:
        if start <= position < end:
            return page, source
    last = page_spans[-1]
    return last[2], last[3]


def _any_ocr_in_range(start, end, page_spans):
    """判断 [start, end) 覆盖到的任一页是否走过 OCR。"""
    for span_start, span_end, _page, _source, ocr_used in page_spans:
        if span_start < end and start < span_end and ocr_used:
            return True
    return False
