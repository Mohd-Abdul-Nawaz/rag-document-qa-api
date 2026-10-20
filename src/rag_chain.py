"""
RAG Chain Module.

Combines vector store retrieval with LLM generation to produce
grounded answers with cited sources. Supports function calling
for metadata filtering and demo mode for running without API keys.
"""

import os
from typing import Any, Dict, List, Optional

from .models import SourceChunk, QueryResponse
from .vector_store import VectorStore
from .ingestion import DocumentIngester

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class RAGChain:
    """
    RAG (Retrieval-Augmented Generation) chain.

    Retrieves relevant document chunks from the vector store and
    uses an LLM to generate an answer grounded in those chunks.
    """

    def __init__(self, vector_store: VectorStore, demo_mode: bool = DEMO_MODE):
        self.vector_store = vector_store
        self.demo_mode = demo_mode
        self.llm = None

        if not demo_mode and OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model="gpt-3.5-turbo",
                    temperature=0,
                )
            except ImportError:
                pass

    async def query(
        self,
        question: str,
        top_k: int = 5,
        use_function_calling: bool = True,
        filter_metadata: Optional[dict] = None,
    ) -> QueryResponse:
        """
        Process a RAG query: retrieve chunks, generate answer, return with sources.

        Args:
            question: User's natural language question.
            top_k: Number of chunks to retrieve.
            use_function_calling: Whether to use LLM function calling.
            filter_metadata: Optional metadata filter for retrieval.

        Returns:
            QueryResponse with answer, sources, and metadata.
        """
        # Step 1: Retrieve relevant chunks
        search_results = self.vector_store.search(
            query=question,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Step 2: Convert to SourceChunk objects
        sources = [
            SourceChunk(
                content=r.get("content", ""),
                document_id=r.get("document_id") or "unknown",
                metadata=r.get("metadata"),
                score=r.get("score", 0.0),
            )
            for r in search_results
        ]

        # Step 3: Generate answer
        if self.demo_mode or self.llm is None:
            answer = self._demo_answer(question, sources)
            function_calls = None
            token_usage = None
        else:
            answer, function_calls, token_usage = await self._generate_answer(
                question, sources, use_function_calling
            )

        return QueryResponse(
            answer=answer,
            sources=sources,
            question=question,
            function_calls=function_calls,
            token_usage=token_usage,
        )

    def _demo_answer(self, question: str, sources: List[SourceChunk]) -> str:
        """Generate a simulated answer for demo mode."""
        if not sources:
            return (
                f"[DEMO] No relevant documents found for: '{question}'. "
                f"Please ingest some documents first."
            )

        context = "\n".join(
            f"[{i+1}] {s.content[:200]}..." for i, s in enumerate(sources[:3])
        )

        return (
            f"[DEMO] Answer to: '{question}'\n\n"
            f"Based on the retrieved document chunks:\n\n{context}\n\n"
            f"---\n"
            f"This is a simulated response. Configure OPENAI_API_KEY for live generation."
        )

    async def _generate_answer(
        self,
        question: str,
        sources: List[SourceChunk],
        use_function_calling: bool,
    ) -> tuple:
        """Generate an answer using the LLM with retrieved context."""
        context = "\n\n".join(
            f"Source {i+1} (doc: {s.document_id}): {s.content}"
            for i, s in enumerate(sources)
        )

        prompt = (
            f"You are a helpful assistant answering questions based on provided documents.\n"
            f"Use only the information from the sources below to answer the question.\n"
            f"Cite sources using [1], [2], etc.\n\n"
            f"Sources:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        try:
            response = await self.llm.ainvoke(prompt)
            answer = response.content
            return answer, None, None
        except Exception as exc:
            return f"[ERROR] Failed to generate answer: {str(exc)}", None, None

    async def ingest_and_index(
        self,
        content: str,
        metadata: Optional[dict] = None,
        document_id: Optional[str] = None,
    ) -> dict:
        """
        Ingest a document, chunk it, and add to the vector store.
        """
        ingester = DocumentIngester(demo_mode=self.demo_mode)
        result = ingester.ingest(
            content=content,
            metadata=metadata,
            document_id=document_id,
        )

        chunk_ids = self.vector_store.add_chunks(result["chunks"])

        return {
            "document_id": result["document_id"],
            "status": "indexed",
            "chunks_created": result["chunks_created"],
            "chunk_ids": chunk_ids,
        }