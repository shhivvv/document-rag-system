import pytest
from unittest.mock import MagicMock, patch
from app.services.contradiction import ContradictionService
from app.services.retriever import DocumentRetriever
from app.services.llm import LLMService
from app.models import DocumentChunk
from app.schemas import ContradictionResponse

@pytest.fixture
def mock_retriever():
    return MagicMock(spec=DocumentRetriever)

@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMService)

@pytest.fixture
def contradiction_service(mock_retriever, mock_llm):
    return ContradictionService(retriever=mock_retriever, llm_service=mock_llm)

def test_analyze_contradiction_success(contradiction_service, mock_retriever, mock_llm):
    # Setup mock chunks
    chunk_doc1 = DocumentChunk(
        id="c1", document_id="d1", text="Employees get 20 days of paid leave.",
        metadata={"filename": "doc1.txt", "page_number": 1}
    )
    chunk_doc2 = DocumentChunk(
        id="c2", document_id="d2", text="Employees get 15 days of paid leave.",
        metadata={"filename": "doc2.txt", "page_number": 1}
    )
    
    mock_retriever.retrieve.side_effect = lambda query, top_k, filename_filter: (
        [chunk_doc1] if filename_filter == "doc1.txt" else [chunk_doc2]
    )

    # Mock JSON response from LLM
    mock_llm.generate.return_value = """
    {
        "status": "Conflict",
        "reasoning": "Document 1 specifies 20 days of leave, while Document 2 specifies 15 days.",
        "document1_evidence": "Employees get 20 days of paid leave.",
        "document2_evidence": "Employees get 15 days of paid leave.",
        "confidence": 0.95
    }
    """
    
    response = contradiction_service.analyze_contradiction("doc1.txt", "doc2.txt", "leave allowance")
    
    assert isinstance(response, ContradictionResponse)
    assert response.status == "Conflict"
    assert response.confidence == 0.95
    assert "20 days" in response.document1_evidence
    assert "15 days" in response.document2_evidence
    assert "specifies 20 days" in response.reasoning

def test_analyze_contradiction_no_relevant_chunks(contradiction_service, mock_retriever):
    mock_retriever.retrieve.return_value = []
    
    with pytest.raises(ValueError) as excinfo:
        contradiction_service.analyze_contradiction("doc1.txt", "doc2.txt", "remote work")
        
    assert "No relevant content found" in str(excinfo.value)
