from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    doc_ids: Optional[List[str]] = Field(default=None)

class SourceChunk(BaseModel):
    doc_id: str
    source: str
    page: Optional[int]
    score: float
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    latency_ms: float

class IngestionResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int