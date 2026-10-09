from config import CHUNK_SIZE
from core.text_splitter import split_text


def test_split_text_returns_chunks_within_config_size():
    chunks = split_text("测试句子。" * 400)

    assert chunks
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)


def test_split_text_accepts_explicit_overrides():
    chunks = split_text("abc " * 200, chunk_size=40, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
