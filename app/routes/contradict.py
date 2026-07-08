import logging
from fastapi import APIRouter, Depends, HTTPException
from app.schemas import ContradictionRequest, ContradictionResponse
from app.dependencies import get_contradiction_service, get_vector_store_service, get_llm_service
from app.services.contradiction import ContradictionService
from app.services.vector_store import VectorStoreService
from app.services.llm import LLMService

router = APIRouter()
logger = logging.getLogger("rag_system.routes.contradict")

@router.post("/contradict", response_model=ContradictionResponse, tags=["Retrieval-QA"])
def check_contradiction(
    request: ContradictionRequest,
    contradiction_service: ContradictionService = Depends(get_contradiction_service),
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Compares two indexed documents on a specific topic.
    Retrieves evidence from both documents and uses the Groq LLM to check if there is a Conflict,
    No Conflict, or Partial Conflict, along with supporting evidence and confidence score.
    """
    # 1. Validation checks
    if not llm.is_configured():
        logger.error("Groq API key is missing or not configured.")
        raise HTTPException(
            status_code=503,
            detail="LLM service is unavailable. Please verify that the GROQ_API_KEY environment variable is configured."
        )

    # 2. Check if documents are ingested in vector database
    available_docs = vector_store.get_all_document_names()
    
    missing_docs = []
    if request.document1 not in available_docs:
        missing_docs.append(request.document1)
    if request.document2 not in available_docs:
        missing_docs.append(request.document2)

    if missing_docs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Required document(s) not found in index: {', '.join(missing_docs)}. "
                f"Available documents: {', '.join(available_docs)}"
            )
        )

    if not request.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="The topic field cannot be empty."
        )

    try:
        # Run comparison analysis
        result = contradiction_service.analyze_contradiction(
            doc1_name=request.document1,
            doc2_name=request.document2,
            topic=request.topic
        )
        return result
        
    except ValueError as ve:
        logger.error(f"Validation error in contradiction checker: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error during contradiction analysis: {str(re)}")
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        logger.exception("Unexpected error occurred in /contradict endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal error occurred: {str(e)}"
        )
