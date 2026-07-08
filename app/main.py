import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import health, upload, ask, contradict
from app.dependencies import get_vector_store_service

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_system")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Starting up Document RAG API system...")
    try:
        # Pre-warm connection to vector store collection
        store = get_vector_store_service()
        store.reload_collection()
        logger.info(f"Vector Database warmed up with {store.get_chunk_count()} chunks.")
    except Exception as e:
        logger.error(f"Failed to initialize vector database at startup: {str(e)}")
    
    yield
    
    # Shutdown actions
    logger.info("Shutting down Document RAG API system...")

app = FastAPI(
    title="Production-Grade Document RAG System",
    description=(
        "FastAPI backend for a complete production-quality Document Q&A RAG system "
        "supporting semantic chunking, multilingual search, citations, and contradiction detection."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit and other frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware for Request Timing and Latency Logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    logger.info(
        f"HTTP {request.method} {request.url.path} "
        f"completed in {process_time:.4f}s with status {response.status_code}"
    )
    return response

# Standard route endpoints
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(ask.router)
app.include_router(contradict.router)

@app.get("/", tags=["System"])
def root_info():
    """
    Returns API root specifications.
    """
    return {
        "title": "Production-Grade Document QA RAG system API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

# Exception Handler to catch unexpected errors and return uniform JSON
@app.exception_handler(Exception)
def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception occurred at {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An internal server error occurred: {str(exc)}"}
    )
