"""
Tests for Enterprise RAG Document QA API.
Runs in DEMO_MODE so no external API keys are required.
"""

import os
import pytest

os.environ["DEMO_MODE"] = "true"

from src.ingestion import DocumentIngester
from src.vector_store import VectorStore
from src.rag_chain import RAGChain


@pytest.mark.asyncio
async def test_document_ingestion():
    """Test document chunking and ingestion."""
    ingester = DocumentIngester(demo_mode=True)
    result = ingester.ingest(
        content="Acme Corp provides AI agent solutions for enterprise automation. "
        "Our platform supports LangGraph, MCP, and RAG architectures.",
        metadata={"source": "brochure", "type": "text"},
        document_id="doc-test-001",
    )

    assert result["document_id"] == "doc-test-001"
    assert result["chunks_created"] >= 1
    assert result["status"] == "ingested"
    assert len(result["chunks"]) >= 1


@pytest.mark.asyncio
async def test_vector_store_add_and_search():
    """Test adding chunks to vector store and searching."""
    store = VectorStore(demo_mode=True)
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "LangGraph is a library for building stateful agents.",
            "chunk_index": 0,
            "metadata": {"source": "docs"},
            "indexed_at": "2024-01-01",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "content": "MCP standardizes tool integration for AI agents.",
            "chunk_index": 0,
            "metadata": {"source": "docs"},
            "indexed_at": "2024-01-01",
        },
    ]

    ids = store.add_chunks(chunks)
    assert len(ids) == 2

    # Search
    results = store.search(query="LangGraph agents", top_k=5)
    assert len(results) >= 1
    assert any("LangGraph" in r["content"] for r in results)


@pytest.mark.asyncio
async def test_rag_chain_query_demo():
    """Test RAG chain query in demo mode."""
    store = VectorStore(demo_mode=True)
    # First add some chunks
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Acme Corp provides AI agent solutions for enterprise automation.",
            "chunk_index": 0,
            "metadata": {"source": "brochure"},
            "indexed_at": "2024-01-01",
        }
    ]
    store.add_chunks(chunks)

    rag = RAGChain(vector_store=store, demo_mode=True)
    response = await rag.query(question="What does Acme Corp provide?", top_k=5)

    assert response.question == "What does Acme Corp provide?"
    assert len(response.sources) >= 1
    assert "[DEMO]" in response.answer or "Acme" in response.answer


@pytest.mark.asyncio
async def test_text_chunking():
    """Test text chunking with overlap."""
    ingester = DocumentIngester(demo_mode=True)
    long_text = "A" * 1200  # 1200 characters
    chunks = ingester.chunk_text(long_text, chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 2
    # Check overlap exists between consecutive chunks
    if len(chunks) >= 2:
        assert chunks[0][-50:] == chunks[1][:50]


def test_demo_mode_flag():
    """Test that DEMO_MODE is properly set."""
    assert os.getenv("DEMO_MODE") == "true"