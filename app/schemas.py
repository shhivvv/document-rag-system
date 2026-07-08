from typing import List, Optional
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What is the company policy on remote work?"})
    use_mmr: bool = Field(default=False, description="Enable Maximal Marginal Relevance retrieval")
    target_language: Optional[str] = Field(default=None, description="Force translation into a specific language")

class Citation(BaseModel):
    source_file: str = Field(..., description="Name of the source document file")
    page: Optional[int] = Field(None, description="Page number where the chunk is located (1-indexed)")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    snippet: str = Field(..., description="Snippet of text that supports the answer")
    similarity_score: Optional[float] = Field(None, description="Vector search match score")

class AskResponse(BaseModel):
    question: str
    answer: str
    language: str
    citations: List[Citation]
    latency_seconds: float = Field(..., description="Time taken to process request")

class ContradictionRequest(BaseModel):
    document1: str = Field(..., description="Filename of the first document (e.g. policy.docx)")
    document2: str = Field(..., description="Filename of the second document (e.g. handbook.pdf)")
    topic: str = Field(..., description="The semantic subject to compare (e.g. maternity leave)")

class ContradictionResponse(BaseModel):
    status: str = Field(..., description="Conflict status: 'Conflict', 'No Conflict', or 'Partial Conflict'")
    reasoning: str = Field(..., description="Detailed analytical justification by the LLM")
    document1_evidence: str = Field(..., description="Textual evidence retrieved from Document 1")
    document2_evidence: str = Field(..., description="Textual evidence retrieved from Document 2")
    confidence: float = Field(..., description="Confidence score of contradiction analysis between 0.0 and 1.0")

class HealthResponse(BaseModel):
    status: str
    groq_api_configured: bool
    embedding_model: str
    total_documents: int
    total_chunks: int
