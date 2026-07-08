import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker

@pytest.fixture
def doc_loader():
    return DocumentLoader()

@pytest.fixture
def doc_chunker():
    return DocumentChunker(chunk_size=700, chunk_overlap=120)

def test_clean_text(doc_loader):
    raw_text = "This   is a    text   with   multiple  spaces.\n\n\n\nAnd multiple newlines."
    cleaned = doc_loader.clean_text(raw_text)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "This is a text" in cleaned

def test_load_txt(doc_loader, tmp_path):
    txt_file = tmp_path / "test_file.txt"
    txt_file.write_text("Hello World. " * 200, encoding="utf-8") # 400 words
    
    pages = doc_loader.load_txt(txt_file)
    assert len(pages) == 1
    assert pages[0]["metadata"]["filename"] == "test_file.txt"
    assert pages[0]["metadata"]["page"] == 1
    assert "Hello World." in pages[0]["text"]

def test_load_markdown(doc_loader, tmp_path):
    md_file = tmp_path / "test_file.md"
    md_file.write_text("# Main Heading\nSome markdown text.\n## Sub Heading\nMore text.", encoding="utf-8")
    
    pages = doc_loader.load_markdown(md_file)
    assert len(pages) == 1
    assert pages[0]["metadata"]["filename"] == "test_file.md"
    assert "# Main Heading" in pages[0]["text"]

def test_chunker_detect_sections(doc_chunker):
    text = "Intro Paragraph.\n\n# Section One\nContent for section one.\n\n## Section Two\nContent for section two."
    sections = doc_chunker._detect_sections(text)
    
    assert len(sections) == 3
    assert sections[0]["title"] == "Introduction"
    assert sections[1]["title"] == "Section One"
    assert sections[2]["title"] == "Section Two"

def test_chunk_document(doc_chunker):
    pages = [
        {
            "text": "# Header Title\nThis is a long sentence that will represent the body text of the document. " * 30,
            "metadata": {"filename": "doc.md", "page": 1}
        }
    ]
    
    chunks = doc_chunker.chunk_document(pages)
    assert len(chunks) > 0
    
    # Assert metadata
    first_chunk = chunks[0]
    assert first_chunk.metadata["filename"] == "doc.md"
    assert first_chunk.metadata["page_number"] == 1
    assert first_chunk.metadata["section_title"] == "Header Title"
    assert "chunk_id" in first_chunk.metadata
    assert first_chunk.document_id == doc_chunker._generate_hash_id("doc.md")
