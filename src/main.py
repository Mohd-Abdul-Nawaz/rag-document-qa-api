"""
FastAPI Application Entry Point for Enterprise RAG Document QA API.

Exposes REST endpoints for:
- Health check
- Document ingestion and indexing
- Document QA queries with RAG
- Document listing and deletion
- Semantic search
"""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    QueryRequest,
    QueryResponse,
    DocumentInfo,
    SearchRequest,
    SearchResponse,
)
from .vector_store import VectorStore
from .rag_chain import RAGChain

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

app = FastAPI(
    title="Enterprise RAG Document QA API",
    description="RAG application for querying enterprise documents with FastAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore(demo_mode=DEMO_MODE)
rag_chain = RAGChain(vector_store=vector_store, demo_mode=DEMO_MODE)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "rag-document-qa-api",
        "demo_mode": DEMO_MODE,
        "vector_store": "demo" if DEMO_MODE else "chroma/faiss",
    }


@app.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    """Ingest and index a document into the vector store."""
    try:
        result = await rag_chain.ingest_and_index(
            content=request.content,
            metadata=request.metadata.model_dump() if request.metadata else None,
            document_id=request.document_id,
        )
        return DocumentIngestResponse(
            document_id=result["document_id"],
            status=result["status"],
            chunks_created=result["chunks_created"],
            message=f"Document {result['document_id']} indexed with {result['chunks_created']} chunks.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/documents/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question against indexed documents using RAG."""
    try:
        response = await rag_chain.query(
            question=request.question,
            top_k=request.top_k,
            use_function_calling=request.use_function_calling,
            filter_metadata=request.filter_metadata,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents/list", response_model=List[DocumentInfo])
async def list_documents():
    """List all indexed documents."""
    docs = vector_store.list_documents()
    return [
        DocumentInfo(
            document_id=d["document_id"],
            metadata=d.get("metadata"),
            chunk_count=d.get("chunk_count", 0),
            indexed_at="",
        )
        for d in docs
    ]


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    success = vector_store.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": document_id}


@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Perform semantic search without generation."""
    results = vector_store.search(
        query=request.query,
        top_k=request.top_k,
        filter_metadata=request.filter_metadata,
    )
    return SearchResponse(
        query=request.query,
        results=[
            {
                "content": r.get("content", ""),
                "document_id": r.get("document_id") or "unknown",
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata"),
            }
            for r in results
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("APP_PORT", "8002")))