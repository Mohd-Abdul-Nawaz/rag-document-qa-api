"""
API smoke tests for the Enterprise RAG Document QA API.

Exercises every REST route exposed by the FastAPI app in demo mode, so a
reviewer can confirm the service works end to end without any API keys:

    cd rag-document-qa-api
    DEMO_MODE=true pytest tests/test_api.py
"""

import os
import pytest
from fastapi.testclient import TestClient

os.environ["DEMO_MODE"] = "true"

from src.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "rag-document-qa-api"


def test_ingest_list_query_search_delete(client):
    ingest = client.post(
        "/documents/ingest",
        json={
            "content": (
                "Agentic AI refers to systems that plan, use tools, and reflect. "
                "LangGraph builds stateful multi-agent workflows with checkpoints. "
                "Model Context Protocol standardises how agents connect to tools."
            ),
            "metadata": {"source": "smoke-test", "document_type": "txt"},
        },
    )
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]
    assert ingest.json()["chunks_created"] >= 1

    listing = client.get("/documents/list")
    assert listing.status_code == 200
    assert document_id in [d["document_id"] for d in listing.json()]

    query = client.post(
        "/documents/query",
        json={"question": "What is LangGraph?", "top_k": 3},
    )
    assert query.status_code == 200
    body = query.json()
    assert body["answer"]
    assert body["sources"]
    # Every source must carry a document_id so answers remain citable.
    for source in body["sources"]:
        assert source["document_id"]
        assert source["document_id"] != "unknown"

    search = client.post("/search", json={"query": "LangGraph", "top_k": 3})
    assert search.status_code == 200
    assert search.json()["results"]
    for result in search.json()["results"]:
        assert result["document_id"]
        assert result["document_id"] != "unknown"

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"


def test_query_with_no_documents_returns_empty_sources(client):
    response = client.post(
        "/documents/query",
        json={"question": "a question with no matching document", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_delete_missing_document_returns_404(client):
    assert client.delete("/documents/does-not-exist").status_code == 404


def test_document_id_propagates_through_vector_store():
    """Chroma-style retrieval must preserve the parent document_id."""
    from src.ingestion import DocumentIngester
    from src.vector_store import VectorStore

    store = VectorStore(demo_mode=True)
    ingester = DocumentIngester(demo_mode=True)

    result = ingester.ingest(
        content="LangGraph enables checkpointed agent state.",
        metadata={"source": "unit-test"},
        document_id="doc-propagation-001",
    )
    store.add_chunks(result["chunks"])

    hits = store.search(query="LangGraph checkpointed state", top_k=3)
    assert hits
    for hit in hits:
        assert hit["document_id"] == "doc-propagation-001"
        assert hit["chunk_id"].startswith("doc-propagation-001-chunk-")


def test_live_chroma_preserves_document_id():
    """
    Round-trip a document through a real Chroma collection.

    Skipped automatically when `chromadb` is not installed. Uses a local
    deterministic embedding function, so no API key or network access is
    required.
    """
    pytest.importorskip("chromadb")

    from src.ingestion import DocumentIngester
    from src.vector_store import VectorStore

    store = VectorStore(demo_mode=False, store_type="chroma")
    if store.demo_mode:
        pytest.skip("Chroma store unavailable in this environment")

    ingester = DocumentIngester(demo_mode=False)
    result = ingester.ingest(
        content=(
            "LangGraph builds checkpointed stateful agents for multi-step workflows. "
            "Model Context Protocol standardises how agents connect to external tools."
        ),
        metadata={"source": "unit-test", "document_type": "txt"},
        document_id="doc-live-chroma-001",
    )
    store.add_chunks(result["chunks"])

    hits = store.search(query="LangGraph checkpointed agents", top_k=3)
    assert hits, "expected at least one Chroma hit"
    for hit in hits:
        assert hit["document_id"] == "doc-live-chroma-001"
        assert hit["chunk_id"].startswith("doc-live-chroma-001-chunk-")
        assert 0.0 <= hit["score"] <= 1.0
