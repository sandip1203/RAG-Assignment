# Conversational RAG API

A modular FastAPI backend for PDF/TXT ingestion and document-grounded chat. It uses
Pinecone for vector search, PostgreSQL for document metadata and interview bookings,
Redis for conversation history, and OpenRouter for chat and embeddings.

## Requirements

- Python 3.11+
- PostgreSQL and Redis
- OpenRouter API key (chat and embeddings) and Pinecone API key
- A Pinecone serverless index, created automatically on first ingestion

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the database, provider, and Pinecone values in `.env`, then run:

```bash
uvicorn app.main:app --reload
```

FastAPI creates the SQL tables during application startup. Interactive API
documentation is available at `http://localhost:8000/docs`.

## API

### Ingest a document

`POST /api/v1/documents/ingest` accepts multipart form data:

- `file`: a `.pdf` or UTF-8 `.txt` file (maximum 20 MiB by default)
- `strategy`: `recursive` (default) or `sentence`

Example:

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@handbook.pdf" \
  -F "strategy=sentence"
```

The extracted text is chunked and embedded, vectors are upserted into Pinecone,
and document metadata is stored in PostgreSQL.

### Chat and book an interview

`POST /api/v1/chat` accepts JSON:

```json
{
  "message": "What is the company's interview process?",
  "conversation_id": "candidate-session-1"
}
```

Reuse `conversation_id` on subsequent requests to continue the Redis-backed
conversation. Omit it to start a new conversation. To book, provide your name,
email, date (`YYYY-MM-DD`), and time (`HH:MM`) over one or more turns. The LLM
extracts booking details from conversation history; complete bookings are
validated and persisted in PostgreSQL.

## Configuration

See `.env.example` for supported environment variables. `OPENROUTER_API_KEY`
authenticates both chat and embeddings; `OPENROUTER_MODEL` configures chat and
`EMBEDDING_MODEL` configures embeddings. The default embedding model is
`openai/text-embedding-3-small`. `PINECONE_INDEX_NAME` names the vector index and
must contain only lowercase letters, digits, or hyphens (maximum 45 characters).
Its dimension must match `EMBEDDING_DIMENSION`.
