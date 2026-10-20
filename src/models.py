"""
Pydantic Models for RAG Document QA API.

Defines request/response schemas for document ingestion, querying,
and search endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""
    source: Optional[str] = None
    document_type: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    tags: Optional[List[str]] = None


class DocumentIngestRequest(BaseModel):
    """Request to ingest a document into the vector store."""
    content: str = Field(..., description="The document text content")
    metadata: Optional[DocumentMetadata] = None
    document_id: Optional[str] = None


class DocumentIngestResponse(BaseModel):
    """Response after document ingestion."""
    document_id: str
    status: str
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    """Request to query indexed documents."""
    question: str = Field(..., description="The natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    use_function_calling: bool = Field(default=True, description="Use LLM function calling for filtering")
    filter_metadata: Optional[Dict[str, Any]] = None


class SourceChunk(BaseModel):
    """A source chunk returned with the answer."""
    content: str
    document_id: str
    metadata: Optional[DocumentMetadata] = None
    score: float


class QueryResponse(BaseModel):
    """Response from document QA query."""
    answer: str
    sources: List[SourceChunk]
    question: str
    function_calls: Optional[List[dict]] = None
    token_usage: Optional[dict] = None


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    document_id: str
    metadata: Optional[DocumentMetadata] = None
    chunk_count: int
    indexed_at: str


class SearchRequest(BaseModel):
    """Request for semantic search (without generation)."""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filter_metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """A single search result."""
    content: str
    document_id: str
    score: float
    metadata: Optional[DocumentMetadata] = None


class SearchResponse(BaseModel):
    """Response from semantic search."""
    results: List[SearchResult]
    query: str