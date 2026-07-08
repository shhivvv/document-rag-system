import logging
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retriever import DocumentRetriever
from app.services.llm import LLMService
from app.services.citation import CitationService
from app.services.translation import TranslationService
from app.services.contradiction import ContradictionService
from app.services.rag_pipeline import RAGPipeline
from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker

logger = logging.getLogger("rag_system.dependencies")

# Create Singletons
logger.info("Initializing global services singletons...")

embedding_service = EmbeddingService()
vector_store_service = VectorStoreService(embedding_service=embedding_service)
retriever_service = DocumentRetriever(vector_store=vector_store_service, embedding_service=embedding_service)
llm_service = LLMService()

citation_service = CitationService()
translation_service = TranslationService(llm_service=llm_service)
contradiction_service = ContradictionService(retriever=retriever_service, llm_service=llm_service)

rag_pipeline = RAGPipeline(
    llm_service=llm_service,
    retriever=retriever_service,
    citation_service=citation_service,
    translation_service=translation_service
)

document_loader = DocumentLoader()
document_chunker = DocumentChunker()

# Dependency providers for FastAPI
def get_embedding_service() -> EmbeddingService:
    return embedding_service

def get_vector_store_service() -> VectorStoreService:
    return vector_store_service

def get_retriever_service() -> DocumentRetriever:
    return retriever_service

def get_llm_service() -> LLMService:
    return llm_service

def get_contradiction_service() -> ContradictionService:
    return contradiction_service

def get_rag_pipeline() -> RAGPipeline:
    return rag_pipeline

def get_document_loader() -> DocumentLoader:
    return document_loader

def get_document_chunker() -> DocumentChunker:
    return document_chunker
