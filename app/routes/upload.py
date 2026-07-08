import logging
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.config import settings
from app.dependencies import (
    get_vector_store_service,
    get_document_loader,
    get_document_chunker
)
from app.services.vector_store import VectorStoreService
from app.services.document_loader import DocumentLoader
from app.services.chunking import DocumentChunker

router = APIRouter()
logger = logging.getLogger("rag_system.routes.upload")

@router.post("/upload", tags=["Ingestion"])
async def upload_file(
    file: UploadFile = File(...),
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    loader: DocumentLoader = Depends(get_document_loader),
    chunker: DocumentChunker = Depends(get_document_chunker)
):
    """
    Ingests an uploaded document (PDF, DOCX, TXT, MD).
    Saves it locally, extracts text, generates semantic chunks, computes embeddings,
    and indexes them in the persistent vector store. Prevents duplicate chunks.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    # Validate file extension
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".docx", ".txt", ".md", ".markdown"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only PDF, DOCX, TXT, and Markdown files are supported."
        )

    # Resolve save path
    upload_path = settings.get_upload_path() / filename
    logger.info(f"Saving uploaded file '{filename}' to '{upload_path}'")

    try:
        # Save file to disk
        with open(upload_path, "wb") as f:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"Uploaded file '{filename}' is empty.")
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to write file to disk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # Parse / Load file
        pages_content = loader.load_document(upload_path)
    except ValueError as ve:
        # Cleanup corrupt/empty saved file
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"Document parsing validation failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse document content: {str(ve)}")
    except Exception as e:
        # Cleanup saved file
        if upload_path.exists():
            upload_path.unlink()
        logger.error(f"Error loading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading document structure: {str(e)}")

    try:
        # Chunk text
        chunks = chunker.chunk_document(pages_content)
        if not chunks:
            raise ValueError("No text chunks generated. The document text might be empty or unparseable.")
    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to chunk document: {str(e)}")

    try:
        # Index in vector store
        ingest_stats = vector_store.add_chunks(chunks)
        # Reload collection to sync
        vector_store.reload_collection()
    except Exception as e:
        logger.error(f"Vector DB indexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector DB indexing failed: {str(e)}")

    return {
        "message": f"Successfully processed '{filename}'",
        "filename": filename,
        "total_chunks_processed": len(chunks),
        "chunks_added": ingest_stats["added"],
        "duplicates_skipped": ingest_stats["skipped"]
    }
