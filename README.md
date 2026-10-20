# Enterprise RAG Document QA API

A Retrieval-Augmented Generation (RAG) application for querying enterprise documents (PDFs, emails, structured data) using natural language. Built with **FastAPI**, **Chroma/FAISS**, and **OpenAI/Azure OpenAI**.

## Features

- **Document ingestion**: Parse, chunk, and index PDFs, text files, and emails
- **Vector storage**: Chroma or FAISS backend for similarity search
- **RAG pipeline**: Retrieve relevant chunks + generate answers with cited sources
- **Function calling**: LLM can trigger metadata filters and document-specific queries
- **Structured outputs**: Pydantic models for request/response validation
- **FastAPI backend**: Async REST API with automatic Swagger UI
- **Unit tests**: pytest-based testing with sample data
- **Docker-ready** for deployment

## Project Structure

```
rag-document-qa-api/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI entry point
│   ├── ingestion.py     # Document loading and chunking
│   ├── vector_store.py  # Chroma/FAISS vector operations
│   ├── rag_chain.py     # RAG chain with function calling
│   └── models.py        # Pydantic request/response models
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   └── sample_data/
│       └── sample.txt
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd rag-document-qa-api
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required environment variables:
- `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` — LLM and embeddings provider
- `VECTOR_STORE_TYPE` — `chroma` or `faiss` (default: chroma)

### 3. Run the Application

```bash
# Option A: Run directly
python -m src.main

# Option B: Run with uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8002

# Option C: Run with Docker
docker build -t rag-document-qa-api .
docker run -p 8002:8002 --env-file .env rag-document-qa-api
```

### 4. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/documents/ingest` | Ingest and index a document |
| POST | `/documents/query` | Ask a question against indexed documents |
| GET | `/documents/list` | List all indexed documents |
| DELETE | `/documents/{doc_id}` | Delete a document |
| GET | `/search` | Semantic search without generation |

### Example: Ingest a Document

```bash
curl -X POST "http://localhost:8002/documents/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Acme Corp provides AI agent solutions for enterprise automation. Our platform supports LangGraph, MCP, and RAG architectures.",
    "metadata": {"source": "company-brochure", "type": "text"},
    "document_id": "doc-001"
  }'
```

### Example: Query Documents

```bash
curl -X POST "http://localhost:8002/documents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What AI agent solutions does Acme Corp provide?",
    "top_k": 5,
    "use_function_calling": true
  }'
```

## Demo Mode

Set `DEMO_MODE=true` in `.env` to run with simulated responses without
requiring OpenAI API keys. Documents are stored in-memory.

## Running Tests

```bash
pytest tests/ -v
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

## License

MIT