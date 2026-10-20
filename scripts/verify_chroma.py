"""
Manual verification script for the live (non-demo) Chroma retrieval path.

Run with:
    python scripts/verify_chroma.py

Confirms that `document_id` and source citations survive a real Chroma
round-trip, without requiring any external API keys.
"""

import os
import sys

os.environ["DEMO_MODE"] = "false"
os.environ["VECTOR_STORE_TYPE"] = "chroma"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import DocumentIngester  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def main() -> int:
    store = VectorStore(demo_mode=False, store_type="chroma")
    print(f"demo_mode={store.demo_mode} store_type={store.store_type}")

    ingester = DocumentIngester(demo_mode=False)
    result = ingester.ingest(
        content=(
            "LangGraph builds checkpointed stateful agents for multi-step workflows. "
            "Model Context Protocol standardises how agents connect to external tools."
        ),
        metadata={"source": "verification", "document_type": "txt"},
        document_id="doc-chroma-verify-001",
    )
    ids = store.add_chunks(result["chunks"])
    print(f"added {len(ids)} chunk(s): {ids}")

    hits = store.search(query="LangGraph checkpointed agents", top_k=3)
    print(f"retrieved {len(hits)} hit(s)")
    for hit in hits:
        print(
            f"  chunk_id={hit['chunk_id']} "
            f"document_id={hit['document_id']} "
            f"score={hit['score']}"
        )
        print(f"  metadata={hit['metadata']}")

    if not hits:
        print("FAIL: no hits returned")
        return 1

    bad = [h for h in hits if h.get("document_id") != "doc-chroma-verify-001"]
    if bad:
        print("FAIL: document_id not propagated")
        return 1

    print("PASS: document_id propagated through Chroma retrieval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
