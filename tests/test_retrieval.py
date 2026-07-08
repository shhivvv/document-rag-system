import pytest
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retriever import DocumentRetriever
from app.models import DocumentChunk
from app.config import settings

@pytest.fixture
def temp_db_dir(tmp_path):
    # Override settings for persist directory
    original_persist_dir = settings.CHROMA_PERSIST_DIR
    temp_dir = tmp_path / "vector_db"
    settings.CHROMA_PERSIST_DIR = str(temp_dir)
    yield temp_dir
    settings.CHROMA_PERSIST_DIR = original_persist_dir

@pytest.fixture
def mock_embedding_service():
    service = MagicMock(spec=EmbeddingService)
    service.model_name = "all-MiniLM-L6-v2"
    # Return dummy vectors of length 384 (MiniLM size)
    service.embed_query.return_value = [0.1] * 384
    service.embed_documents.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
    return service

@pytest.fixture
def vector_store(temp_db_dir, mock_embedding_service):
    store = VectorStoreService(embedding_service=mock_embedding_service)
    # Recreate collection to start clean
    store.delete_collection()
    return store

@pytest.fixture
def retriever(vector_store, mock_embedding_service):
    return DocumentRetriever(vector_store=vector_store, embedding_service=mock_embedding_service)

def test_add_chunks_and_duplicates(vector_store):
    chunks = [
        DocumentChunk(
            id="chunk_1",
            document_id="doc_1",
            text="This is unique text for chunk one.",
            metadata={"filename": "doc1.txt", "page_number": 1, "section_title": "Intro"}
        ),
        DocumentChunk(
            id="chunk_2",
            document_id="doc_1",
            text="This is unique text for chunk two.",
            metadata={"filename": "doc1.txt", "page_number": 1, "section_title": "Body"}
        )
    ]
    
    # First Ingest
    res = vector_store.add_chunks(chunks)
    assert res["added"] == 2
    assert res["skipped"] == 0
    assert vector_store.get_chunk_count() == 2

    # Second Ingest (Duplicate Check)
    res_dup = vector_store.add_chunks(chunks)
    assert res_dup["added"] == 0
    assert res_dup["skipped"] == 2
    assert vector_store.get_chunk_count() == 2

def test_retriever_similarity_search(retriever, vector_store):
    chunks = [
        DocumentChunk(
            id="chunk_a",
            document_id="doc_a",
            text="Machine learning is a field of artificial intelligence.",
            metadata={"filename": "ml.txt", "page_number": 1, "section_title": "AI"}
        )
    ]
    vector_store.add_chunks(chunks)
    
    results = retriever.retrieve(query="machine learning", top_k=1, use_mmr=False)
    assert len(results) == 1
    assert results[0].id == "chunk_a"
    assert "artificial intelligence" in results[0].text
    assert results[0].score is not None

def test_cosine_similarity(retriever):
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    
    assert retriever._cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert retriever._cosine_similarity(v1, v3) == pytest.approx(0.0)

def test_mmr_selection(retriever):
    query_vector = [1.0, 0.0]
    
    # Candidate 1: identical to query
    c1_emb = [1.0, 0.0]
    c1_chunk = DocumentChunk(id="c1", document_id="d", text="C1", metadata={})
    
    # Candidate 2: different direction, moderately similar
    c2_emb = [0.8, 0.6]
    c2_chunk = DocumentChunk(id="c2", document_id="d", text="C2", metadata={})
    
    # Candidate 3: redundant with c1 (very close)
    c3_emb = [0.99, 0.05]
    c3_chunk = DocumentChunk(id="c3", document_id="d", text="C3", metadata={})

    candidates = [c1_chunk, c2_chunk, c3_chunk]
    embeddings = [c1_emb, c2_emb, c3_emb]
    
    # We want top 2
    selected = retriever._maximal_marginal_relevance(
        query_vector=query_vector,
        candidate_embeddings=embeddings,
        candidate_chunks=candidates,
        top_k=2,
        lambda_val=0.3
    )
    
    assert len(selected) == 2
    # First should be the most relevant (c1)
    assert selected[0].id == "c1"
    # Second should be c2 rather than c3 because c3 is redundant with c1 (very close embeddings)
    assert selected[1].id == "c2"
