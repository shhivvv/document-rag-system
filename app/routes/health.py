from fastapi import APIRouter, Depends
from app.schemas import HealthResponse
from app.dependencies import get_vector_store_service, get_llm_service
from app.services.vector_store import VectorStoreService
from app.services.llm import LLMService

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Returns application health indicators, including the status of vector store
    indexes and validation of Groq API configuration.
    """
    try:
        total_chunks = vector_store.get_chunk_count()
        total_docs = vector_store.get_document_count()
        status = "healthy"
    except Exception:
        total_chunks = 0
        total_docs = 0
        status = "unhealthy"

    return HealthResponse(
        status=status,
        groq_api_configured=llm.is_configured(),
        embedding_model=vector_store.embedding_service.model_name,
        total_documents=total_docs,
        total_chunks=total_chunks
    )
