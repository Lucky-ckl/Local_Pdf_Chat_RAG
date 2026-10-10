from config import CHUNK_SIZE
from core.document_loader import Block
from core.text_splitter import Chunk, split_blocks, split_text


def test_split_text_returns_chunks_within_config_size():
    chunks = split_text("测试句子。" * 400)

    assert chunks
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)


def test_split_text_accepts_explicit_overrides():
    chunks = split_text("abc " * 200, chunk_size=40, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_split_blocks_assigns_page_and_chunk_index():
    blocks = [
        Block(text="A" * 200, page=1, source="a.pdf", ocr_used=False),
        Block(text="B" * 200, page=2, source="a.pdf", ocr_used=False),
    ]

    chunks = split_blocks(blocks, chunk_size=100, chunk_overlap=0)

    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.source == "a.pdf" for chunk in chunks)
    assert all(chunk.page in (1, 2) for chunk in chunks)
    pages = [chunk.page for chunk in chunks]
    assert pages == sorted(pages)  # 按全文顺序切分，页码单调不减


def test_split_blocks_non_pdf_uses_page_zero():
    blocks = [Block(text="hello world " * 50, page=0, source="t.txt", ocr_used=False)]

    chunks = split_blocks(blocks, chunk_size=60, chunk_overlap=0)

    assert chunks
    assert all(chunk.page == 0 for chunk in chunks)
    assert all(chunk.source == "t.txt" for chunk in chunks)


def test_split_blocks_marks_ocr_used_when_page_used_ocr():
    blocks = [Block(text="X" * 200, page=1, source="s.pdf", ocr_used=True)]

    chunks = split_blocks(blocks, chunk_size=100, chunk_overlap=0)

    assert chunks
    assert all(chunk.ocr_used is True for chunk in chunks)


def test_split_blocks_cross_page_chunk_takes_start_page():
    blocks = [
        Block(text="AAA", page=1, source="x.pdf", ocr_used=False),
        Block(text="BBB" * 100, page=2, source="x.pdf", ocr_used=False),
    ]

    chunks = split_blocks(blocks, chunk_size=120, chunk_overlap=0)

    assert chunks[0].page == 1  # 跨页 chunk 记录起始页
    assert chunks[0].text.startswith("AAA")
