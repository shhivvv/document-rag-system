import logging
import time
from fastapi import APIRouter, Depends, HTTPException
from app.schemas import AskRequest, AskResponse
from app.dependencies import get_rag_pipeline, get_llm_service
from app.services.rag_pipeline import RAGPipeline
from app.services.llm import LLMService

router = APIRouter()
logger = logging.getLogger("rag_system.routes.ask")

@router.post("/ask", response_model=AskResponse, tags=["Retrieval-QA"])
def ask_question(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    llm: LLMService = Depends(get_llm_service)
):
    """
    Answers a question based on retrieved document chunks.
    Automatically detects input language, translates query if needed,
    queries ChromaDB, generates facts-only answer, extracts citations,
    and translates the final output back to the original language.
    """
    start_time = time.time()
    
    # 1. Validation checks
    if not llm.is_configured():
        logger.error("Groq API key is missing or not configured.")
        raise HTTPException(
            status_code=503,
            detail="LLM service is unavailable. Please verify that the GROQ_API_KEY environment variable is configured."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="The question field cannot be empty."
        )

    try:
        # Run RAG execution pipeline
        response = pipeline.answer_question(request)
        
        # Log response latency as requested in extras
        latency = time.time() - start_time
        logger.info(f"Successfully processed query in {latency:.4f} seconds.")
        
        return response
        
    except ValueError as ve:
        logger.error(f"Validation error in QA process: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error during LLM generation: {str(re)}")
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        logger.exception("Unexpected error occurred in /ask endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal error occurred: {str(e)}"
        )
