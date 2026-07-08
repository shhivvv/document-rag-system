from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    """
    Represents a single chunk of a document stored in the vector database
    or retrieved from it.
    """
    id: str = Field(..., description="Unique hash representing the chunk content and metadata")
    document_id: str = Field(..., description="ID of the parent document")
    text: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (page, filename, headers, index)")
    score: Optional[float] = Field(None, description="Similarity score, if retrieved via query")

class DocumentMetadata(BaseModel):
    """
    Represents metadata parsed from a document.
    """
    filename: str
    document_id: str
    total_chunks: int
    file_size_bytes: int
    mime_type: str
