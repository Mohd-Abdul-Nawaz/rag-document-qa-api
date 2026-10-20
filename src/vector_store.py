"""
Vector Store Module.

Provides a unified interface for Chroma and FAISS vector stores
with demo mode fallback using in-memory storage.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()


class VectorStore:
    """
    Vector store for document embeddings and similarity search.

    Supports Chroma, FAISS, and in-memory demo mode.
    """

    def __init__(self, demo_mode: bool = DEMO_MODE, store_type: str = VECTOR_STORE_TYPE):
        self.demo_mode = demo_mode
        self.store_type = store_type
        self._documents: Dict[str, dict] = {}
        self._chunks: Dict[str, dict] = {}
        self.embedder = None
        self.index = None
        self.id_map: Dict[int, str] = {}

        if not demo_mode:
            self._init_production_store()

    def _init_production_store(self):
        """Initialise the configured vector store and (optional) embedder."""
        try:
            self._init_embedder()
        except Exception:
            self.embedder = None

        try:
            if self.store_type == "chroma":
                import chromadb

                self.client = chromadb.Client()
                self.collection = self.client.get_or_create_collection(
                    "documents",
                    embedding_function=self._chroma_embedding_function(),
                )
            elif self.store_type == "faiss":
                import faiss

                dim = 1536
                if self.embedder is not None:
                    dim = self.embedder.dimension
                self.index = faiss.IndexFlatL2(dim)
                self.id_map = {}
            else:
                self.demo_mode = True
        except ImportError:
            self.demo_mode = True
        except Exception:
            self.demo_mode = True

    def _chroma_embedding_function(self):
        """
        Build the embedding function passed to the Chroma collection.

        Chroma's default embedding function downloads an ONNX model on first
        use, which fails without network access. We therefore use the
        configured LangChain/OpenAI embedder when available and otherwise fall
        back to a deterministic local hashing embedder so the service still
        works fully offline.
        """
        if self.embedder is not None:
            try:
                from chromadb.utils.embedding_functions import (
                    ChromaLangchainEmbeddingFunction,
                )

                return ChromaLangchainEmbeddingFunction(self.embedder)
            except Exception:
                pass
        try:
            from chromadb.api.types import EmbeddingFunction, Documents
            from chromadb.utils.embedding_functions import register_embedding_function

            @register_embedding_function
            class LocalHashingEmbeddingFunction(EmbeddingFunction[Documents]):
                """Deterministic, dependency-free embedding for offline use."""

                def __init__(self, dimension: int = 384, **kwargs):
                    self.dimension = dimension

                @classmethod
                def name(cls) -> str:
                    return "local-hashing"

                @classmethod
                def build_from_config(cls, config):
                    return cls(**(config or {}))

                def get_config(self):
                    return {"dimension": self.dimension}

                def __call__(self, input):
                    import hashlib
                    import math

                    if isinstance(input, str):
                        input = [input]
                    vectors = []
                    for text in input:
                        vec = [0.0] * self.dimension
                        tokens = (text or "").lower().split()
                        for token in tokens:
                            digest = hashlib.sha256(token.encode("utf-8")).digest()
                            idx = int.from_bytes(digest[:4], "big") % self.dimension
                            sign = 1.0 if digest[4] % 2 == 0 else -1.0
                            vec[idx] += sign
                        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                        vectors.append([v / norm for v in vec])
                    return vectors

            return LocalHashingEmbeddingFunction()
        except Exception:
            return None

    def _init_embedder(self):
        """
        Create an OpenAI embedding client when a key is configured.

        Without an API key the vector store still works, but FAISS falls back
        to lexical matching instead of true vector similarity.
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return
        try:
            from langchain_openai import OpenAIEmbeddings

            self.embedder = OpenAIEmbeddings(
                api_key=api_key, model="text-embedding-3-small"
            )
        except ImportError:
            self.embedder = None
        except Exception:
            self.embedder = None

    def add_chunks(self, chunks: List[dict]) -> List[str]:
        if self.demo_mode:
            ids = []
            for chunk in chunks:
                cid = chunk["chunk_id"]
                self._chunks[cid] = chunk
                self._documents.setdefault(chunk["document_id"], []).append(cid)
                ids.append(cid)
            return ids
        try:
            if self.store_type == "chroma":
                metadatas = []
                for c in chunks:
                    meta = dict(c.get("metadata") or {})
                    # Chroma only stores scalar metadata, and we need the parent
                    # document id available at query time.
                    meta["document_id"] = c["document_id"]
                    meta["chunk_index"] = int(c.get("chunk_index", 0))
                    metadatas.append(meta)
                self.collection.add(
                    ids=[c["chunk_id"] for c in chunks],
                    documents=[c["content"] for c in chunks],
                    metadatas=metadatas,
                )
                return [c["chunk_id"] for c in chunks]
            elif self.store_type == "faiss":
                import numpy as np

                ids = []
                texts = []
                for chunk in chunks:
                    cid = chunk["chunk_id"]
                    self._chunks[cid] = chunk
                    self._documents.setdefault(chunk["document_id"], []).append(cid)
                    ids.append(cid)
                    texts.append(chunk["content"])

                if self.embedder is not None and texts:
                    vectors = np.array(self.embedder.embed_documents(texts), dtype="float32")
                    self.index.add(vectors)
                    for offset, cid in enumerate(ids):
                        self.id_map[self.index.ntotal - len(ids) + offset] = cid
                return ids
        except Exception:
            return []

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[dict]:
        if self.demo_mode:
            results = []
            query_lower = query.lower()
            for cid, chunk in self._chunks.items():
                content = chunk["content"].lower()
                score = sum(
                    1 for word in query_lower.split() if word in content
                ) / max(len(query_lower.split()), 1)
                if score > 0 or len(results) < top_k:
                    results.append(
                        {
                            "chunk_id": cid,
                            "document_id": chunk["document_id"],
                            "content": chunk["content"],
                            "metadata": chunk.get("metadata", {}),
                            "score": round(min(score + 0.5, 0.99), 3),
                        }
                    )
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        try:
            if self.store_type == "chroma":
                where = filter_metadata or {}
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where if where else None,
                )
                formatted = []
                ids = results.get("ids", [[]])[0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                for i, chunk_id in enumerate(ids):
                    metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                    # Chroma stores `document_id` inside the metadata payload, so
                    # surface it explicitly to keep the response contract stable.
                    document_id = metadata.get("document_id") or self._document_id_for_chunk(
                        chunk_id
                    )
                    # Chroma returns a distance (lower is better); convert it to a
                    # similarity score in [0, 1] to match the demo-mode contract.
                    distance = float(distances[i]) if i < len(distances) else 0.5
                    formatted.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "content": documents[i] if i < len(documents) else "",
                            "metadata": metadata,
                            "score": round(1.0 / (1.0 + distance), 3),
                        }
                    )
                return formatted
            elif self.store_type == "faiss":
                return self._faiss_search(query, top_k)
        except Exception:
            return []

    def _document_id_for_chunk(self, chunk_id: str) -> str:
        """Derive the parent document_id from a chunk_id (`<doc_id>-chunk-<n>`)."""
        if "-chunk-" in chunk_id:
            return chunk_id.rsplit("-chunk-", 1)[0]
        return chunk_id

    def _faiss_search(self, query: str, top_k: int) -> List[dict]:
        """
        Search the FAISS index.

        FAISS stores raw embedding vectors, so this requires an embedding
        function to project the query into the same vector space. When no
        embedder is configured we fall back to a lexical scan over the
        locally cached chunk content so the endpoint still returns useful
        (if approximate) results instead of failing.
        """
        if self.embedder is None:
            return self._lexical_search(query, top_k)

        vector = self.embedder.embed_query(query)
        scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self.id_map.get(int(idx))
            if not chunk_id:
                continue
            chunk = self._chunks.get(chunk_id, {})
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id", self._document_id_for_chunk(chunk_id)),
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {}),
                    "score": round(float(1.0 / (1.0 + float(score))), 3),
                }
            )
        return results

    def _lexical_search(self, query: str, top_k: int) -> List[dict]:
        """Keyword-overlap search used when no embedding model is available."""
        query_words = [w for w in query.lower().split() if w]
        if not query_words:
            return []
        scored = []
        for cid, chunk in self._chunks.items():
            content = chunk["content"].lower()
            hits = sum(1 for word in query_words if word in content)
            if hits:
                scored.append(
                    {
                        "chunk_id": cid,
                        "document_id": chunk.get("document_id", self._document_id_for_chunk(cid)),
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {}),
                        "score": round(hits / len(query_words), 3),
                    }
                )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_document_chunks(self, document_id: str) -> List[dict]:
        if self.demo_mode:
            return self._documents.get(document_id, [])
        return []

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all of its chunks. Returns False if unknown."""
        if self.demo_mode:
            chunk_ids = self._documents.pop(document_id, None)
            if chunk_ids is None:
                return False
            for cid in chunk_ids:
                self._chunks.pop(cid, None)
            return True
        return False

    def list_documents(self) -> List[dict]:
        if self.demo_mode:
            docs = []
            for doc_id, chunk_ids in self._documents.items():
                if chunk_ids:
                    first_chunk = self._chunks.get(chunk_ids[0], {})
                    docs.append(
                        {
                            "document_id": doc_id,
                            "metadata": first_chunk.get("metadata", {}),
                            "chunk_count": len(chunk_ids),
                        }
                    )
            return docs
        return []