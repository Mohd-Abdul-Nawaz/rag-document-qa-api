"""
Document Ingestion Module.

Handles loading, chunking, and preprocessing of documents before
they are added to the vector store. Supports demo mode for
running without external dependencies.
"""

import os
import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import DocumentMetadata

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


class DocumentIngester:
    """
    Ingests documents by chunking text and preparing metadata
    for vector store indexing.
    """

    def __init__(self, demo_mode: bool = DEMO_MODE):
        self.demo_mode = demo_mode

    def generate_document_id(self, content: str, metadata: Optional[dict] = None) -> str:
        """Generate a deterministic document ID from content."""
        hash_input = content + str(metadata or {})
        return f"doc-{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"

    def chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: The document text to chunk.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start = end - chunk_overlap

        return chunks

    def ingest(
        self,
        content: str,
        metadata: Optional[DocumentMetadata] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document: generate ID, chunk content, prepare chunks for indexing.

        Returns a dictionary with document_id, chunks, and metadata.
        """
        if document_id is None:
            document_id = self.generate_document_id(content, metadata)

        chunks = self.chunk_text(content)

        chunk_records = []
        for i, chunk in enumerate(chunks):
            if hasattr(metadata, "model_dump"):
                metadata_dict = metadata.model_dump()
            elif hasattr(metadata, "dict"):
                metadata_dict = metadata.dict()
            else:
                metadata_dict = metadata or {}
            chunk_records.append(
                {
                    "chunk_id": f"{document_id}-chunk-{i}",
                    "document_id": document_id,
                    "content": chunk,
                    "chunk_index": i,
                    "metadata": metadata_dict,
                    "indexed_at": datetime.utcnow().isoformat(),
                }
            )

        return {
            "document_id": document_id,
            "chunks": chunk_records,
            "chunks_created": len(chunk_records),
            "status": "ingested",
        }


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.

    In demo mode, returns a placeholder string.
    """
    if DEMO_MODE:
        return "[DEMO] Extracted PDF content placeholder."

    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except ImportError:
        return "[DEMO] pypdf not installed, using placeholder content."
    except Exception as e:
        return f"[ERROR] Failed to extract PDF: {str(e)}"


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] Failed to read text file: {str(e)}"