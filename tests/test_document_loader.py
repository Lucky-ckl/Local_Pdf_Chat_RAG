from pathlib import Path

from core.document_loader import extract_text


def test_extract_utf8_text_and_markdown(tmp_path: Path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("RAG combines retrieval and generation.\n中文内容。", encoding="utf-8")

    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text("# Heading\n\nHybrid retrieval", encoding="utf-8")

    assert "中文内容" in extract_text(str(text_file))
    assert "Hybrid retrieval" in extract_text(str(markdown_file))


def test_unsupported_extension_returns_empty_string(tmp_path: Path):
    unsupported = tmp_path / "sample.bin"
    unsupported.write_bytes(b"not a supported document")

    assert extract_text(str(unsupported)) == ""


def test_extract_docx_paragraphs(tmp_path: Path):
    from docx import Document

    docx_file = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("RAG docx paragraph")
    document.save(str(docx_file))

    assert "RAG docx paragraph" in extract_text(str(docx_file))


def test_extract_xlsx_sheets(tmp_path: Path):
    import pandas as pd

    xlsx_file = tmp_path / "sample.xlsx"
    pd.DataFrame({"col": ["value1", "value2"]}).to_excel(xlsx_file, index=False, sheet_name="Sheet1")

    text = extract_text(str(xlsx_file))
    assert "Sheet1" in text
    assert "value1" in text


def test_extract_pptx_text(tmp_path: Path):
    from pptx import Presentation

    pptx_file = tmp_path / "sample.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "RAG pptx title"
    presentation.save(str(pptx_file))

    assert "RAG pptx title" in extract_text(str(pptx_file))
