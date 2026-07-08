import pytest
from app.services.citation import CitationService
from app.models import DocumentChunk

def test_extract_citations():
    # Setup retrieved chunks
    chunks = [
        DocumentChunk(
            id="chunk_first",
            document_id="doc_1",
            text="The quick brown fox jumps over the lazy dog.",
            metadata={"filename": "fox.txt", "page_number": 12, "section_title": "Animals"},
            score=0.95
        ),
        DocumentChunk(
            id="chunk_second",
            document_id="doc_2",
            text="Artificial intelligence is transforming industries.",
            metadata={"filename": "ai.txt", "page_number": 3, "section_title": "Technology"},
            score=0.88
        )
    ]
    
    # LLM answer citing both chunks
    answer = "The fox jumped over the dog [1] while AI systems are changing the world [2]. And an invalid citation [3]."
    
    citations = CitationService.extract_citations(answer, chunks)
    
    assert len(citations) == 2
    
    # First citation asserts
    assert citations[0].source_file == "fox.txt"
    assert citations[0].page == 12
    assert citations[0].chunk_id == "chunk_first"
    assert "quick brown fox" in citations[0].snippet
    assert citations[0].similarity_score == 0.95
    
    # Second citation asserts
    assert citations[1].source_file == "ai.txt"
    assert citations[1].page == 3
    assert citations[1].chunk_id == "chunk_second"
    assert "Artificial intelligence" in citations[1].snippet
    assert citations[1].similarity_score == 0.88

def test_postprocess_citations_in_text():
    chunks = [
        DocumentChunk(id="c1", document_id="d1", text="Text 1", metadata={}),
        DocumentChunk(id="c2", document_id="d2", text="Text 2", metadata={})
    ]
    
    answer = "Claim 1 [1] and Claim 2 [2] and invalid Claim [3] and malformed [abc]."
    citations = CitationService.extract_citations(answer, chunks)
    
    cleaned = CitationService.postprocess_citations_in_text(answer, citations, chunks)
    
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert "[3]" not in cleaned
    assert "[abc]" in cleaned # non-digit markers left untouched by pattern
