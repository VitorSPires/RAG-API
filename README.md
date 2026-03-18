# RAG-API – FastAPI Vector Database API with PostgreSQL + pgvector + OpenAI

RAG-API is a FastAPI-based backend that implements a complete Retrieval-Augmented Generation (RAG) pipeline.  
It lets you upload documents, split them into text chunks, create embeddings with OpenAI, store everything in PostgreSQL + pgvector, and query the content through semantic search endpoints.

This project demonstrates a complete RAG pipeline built on top of FastAPI, PostgreSQL + pgvector and OpenAI.

## Features

- **Document management**
  - Upload documents (e.g. PDF, DOCX, text) via `/documents/upload`.
  - Store metadata (file id, name, hash, URL, timestamps) in the `documents` table.
  - List, update and delete documents via REST endpoints.

- **Chunking and embeddings**
  - Process documents into text chunks with overlap and token counts.
  - Store chunks in the `text_chunks` table.
  - Create vector embeddings using OpenAI (`text-embedding-3-small`).
  - Store embeddings in PostgreSQL using the `pgvector` extension (`vector(1536)`).

- **Semantic search**
  - `/search`: semantic search with similarity scores and context (previous/next chunk).
  - `/search/llm`: lightweight endpoint optimized for LLMs (text + file metadata only).

- **Statistics**
  - `/documents/stats`: high-level document counts and sizes.
  - `/chunks/stats`: statistics about chunks and token distribution.
  - `/embeddings/stats`: statistics about embeddings and models used.

- **Admin HTML interface**
  - Simple HTML dashboard at `/` or `/interface` to:
    - View document stats.
    - Upload new documents.
    - Open files in the browser.
    - Delete documents and their related data.

---

## Architecture Overview

At a high level, the flow is:

1. A client uploads a document to the API.
2. The document is stored (e.g. in Firebase Storage) and its metadata goes into PostgreSQL.
3. The document is processed into text chunks and stored in the `text_chunks` table.
4. Each chunk receives an embedding using OpenAI and is stored in the `embeddings` table (with `pgvector`).
5. Semantic search endpoints query the vector store to find the most relevant chunks.

<!-- IMAGE_PROMPT:
Generate a clean architecture diagram for this RAG API:
- FastAPI backend (RAG-API)
- Document upload -> document processing (chunking) -> embeddings creation with OpenAI
- PostgreSQL + pgvector as the vector store (documents, text_chunks, embeddings tables)
- Semantic search endpoints (/search, /search/llm) querying the vector store
Style: minimal, modern, developer-friendly, neutral colors, no background image.
-->

<!-- When the image is generated, replace this comment with:
![RAG-API Architecture](docs/images/rag-architecture.png)
-->

---

## Tech Stack

- **Language**: Python 3
- **Web framework**: FastAPI
- **Database**: PostgreSQL
- **Vector extension**: pgvector (`CREATE EXTENSION vector`)
- **ORM / DB access**: SQLAlchemy + `pgvector` integration
- **Embeddings**: OpenAI (`text-embedding-3-small`)
- **Storage (optional)**: Firebase Storage / Google Cloud Storage
- **Server**: Uvicorn (with `Procfile` for Railway)

---

## Getting Started

### Prerequisites

- Python **3.9+**
- PostgreSQL instance where you can install **pgvector**
- OpenAI API key
- (Optional) Firebase project and credentials if you want to store files in Firebase

### Clone the repository

```bash
git clone <your-repo-url>
cd RAG-API
```

### Create and activate a virtual environment

**Windows (PowerShell):**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> Note: Versions are not pinned on purpose. `pip` will install the latest stable versions compatible with this project.

---

## Configuration

### Environment variables

1. Copy the example file and create your own `.env`:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in:

   - PostgreSQL connection URLs (`DATABASE_PUBLIC_URL`, optionally `DATABASE_URL`).
   - `OPENAI_API_KEY`.
   - Firebase / Google Cloud credentials if you plan to upload files there.
   - Optional `HOST` and `PORT` overrides.

> **Security note:**  
> `.env` **must not** be committed to git. If any real key was ever committed in this repo or shared elsewhere, rotate/revoke it immediately in the corresponding provider (OpenAI, Firebase, Railway, etc.).

---

## Database Setup (PostgreSQL + pgvector)

This project uses PostgreSQL as the main data store and the `pgvector` extension to handle embeddings.

### 1. Enable pgvector

Run the script `database/001_install_pgvector.sql` against your PostgreSQL database:

```sql
-- database/001_install_pgvector.sql
CREATE EXTENSION IF NOT EXISTS vector;

SELECT * FROM pg_extension WHERE extname = 'vector';
```

This will install the `vector` type if it is not already available.

### 2. Create tables and indexes

Then run `database/002_tables_creation.sql`:

- Drops existing tables (if they exist).
- Creates the following tables:
  - **`documents`**
    - `id`: primary key.
    - `file_id`: stable ID for the file (often derived from the hash).
    - `file_name`: original file name.
    - `file_hash`: SHA-256 hash of the file content.
    - `file_url`: public URL pointing to the stored file (e.g. Firebase).
    - `processed_at`: timestamp when the document was processed.
  - **`text_chunks`**
    - `id`: primary key.
    - `chunk_id`: unique identifier per chunk (used to join with embeddings).
    - `document_id`: foreign key to `documents.id`.
    - `content`: main text chunk.
    - `overlap_content`: optional overlap region with previous chunk.
    - `chunk_index`: sequential index per document.
    - `token_count`: token count (approximate) for this chunk.
    - `start_position` / `end_position`: character offsets in the original text.
    - `created_at`: creation timestamp.
  - **`embeddings`**
    - `id`: primary key.
    - `chunk_id`: foreign key to `text_chunks.chunk_id`.
    - `embedding_vector`: vector(1536) embedding (for `text-embedding-3-small`).
    - `model_name`: stored model name (e.g. `text-embedding-3-small`).
    - `created_at`, `updated_at`: timestamps.

- Creates useful indexes:
  - On `text_chunks.document_id`, `text_chunks.chunk_id`, `text_chunks.token_count`.
  - On `embeddings.chunk_id`.

This schema is intentionally simple and optimized for:

- Efficient lookups by document id or chunk id.
- Fast vector similarity queries using `pgvector`.

<!-- IMAGE_PROMPT:
Generate a simple relational diagram showing three tables:
- documents (id, file_id, file_name, file_hash, file_url, processed_at)
- text_chunks (id, chunk_id, document_id -> documents.id, content, overlap_content, chunk_index, token_count, start_position, end_position, created_at)
- embeddings (id, chunk_id -> text_chunks.chunk_id, embedding_vector vector(1536), model_name, created_at, updated_at)
Show the relationships between them and keep the style clean and minimal.
-->

<!-- When the image is generated, replace this comment with:
![RAG-API Database Schema](docs/images/rag-database-schema.png)
-->

---

## Related Projects

This API can be used on its own, but it is also designed to work well with:

- **React client application**  
  A frontend where end users can upload and manage documents (files tab) and, optionally, chat with an assistant powered by this RAG backend.

- **Agent / RAG client (e.g. LangChain)**  
  A separate service or script that consumes the `/search` and `/search/llm` endpoints, builds prompts, and answers user questions using the retrieved context.

You can keep the React client and the agent code in their own repositories, pointing both to this API as the shared RAG backend.

---

## Running the API

With your virtual environment active and `.env` configured:

```bash
python main.py
```

Or, directly with Uvicorn:

```bash
uvicorn main:app --reload
```

By default, the API will be available at:

- `http://localhost:8000`

### Useful URLs

- **Admin HTML interface**:  
  - `http://localhost:8000/`
  - or `http://localhost:8000/interface`
- **Swagger UI**:  
  - `http://localhost:8000/docs`
- **ReDoc**:  
  - `http://localhost:8000/redoc`

---

## API Overview

### Health and diagnostics

- `GET /health`  
  Basic health check for the API and database connectivity.

- `GET /test-db`  
  Tests database connectivity, PostgreSQL version, pgvector installation and a small vector operation.

### Document management

- `GET /documents`  
  List all documents and high-level stats (used by the HTML interface).

- `GET /documents/{id}`  
  Get a single document by id.

- `POST /documents/upload`  
  Upload a new document file. The API:
  - Stores the document.
  - Creates metadata in `documents`.
  - Splits it into chunks (`text_chunks`).
  - Creates embeddings (`embeddings`).

- `PUT /documents/{id}`  
  Update document metadata (e.g. `file_name`, `file_url`).

- `DELETE /documents/{id}`  
  Delete a document, its chunks and related embeddings, and remove the file from storage when applicable.

- `GET /documents/stats`  
  High-level statistics about documents (count, total size, last upload date).

- `GET /documents/{id}/chunks`  
  Retrieve all chunks for a given document.

### Chunk and embedding stats

- `GET /chunks/stats`  
  Aggregate statistics about chunks (total, min/avg/max tokens, total tokens).

- `GET /embeddings/stats`  
  Aggregate statistics about embeddings (count, unique chunks, model name).

### Semantic search

- `GET /search`  
  Full semantic search endpoint. Parameters:
  - `query` (string, required): text to search for.
  - `limit` (int, default 5): maximum number of results (1–20).
  - `similarity_threshold` (float, default 0.0): minimum similarity threshold (0.0–1.0).

  Returns:
  - Similar chunks with similarity scores and a bit of surrounding context (previous/next chunk).

- `GET /search/llm`  
  Lightweight semantic search endpoint for LLMs. Parameters:
  - `query` (string, required)
  - `limit` (int, default 5)

  Returns:
  - `query`
  - `sources`: array of `{ text, file_name, file_url }`, where `text` already concatenates previous, current and next chunk.

---

## Deployment (Procfile-compatible platforms)

This repository includes a `Procfile` that can be used on platforms that support the Procfile convention (for example, Railway or Heroku):

```Procfile
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

On a Procfile-compatible PaaS (e.g. Railway):

- The platform can detect the `Procfile`.
- It will start the web process using Uvicorn bound to `0.0.0.0` and the dynamic `$PORT`.

Environment variables (database URLs, OpenAI key, Firebase credentials, etc.) must be configured in the provider settings (Railway, Heroku, etc.).

> You can safely keep the `Procfile` under version control.  
> It does not contain secrets and documents how to run the app in production.

---

## Contributing

This project is primarily a showcase, but contributions are welcome:

- Open issues for bugs or ideas.
- Submit pull requests with clear descriptions and small, focused changes.
- Follow the existing code style (FastAPI + SQLAlchemy, English comments, no secrets in code).

---

## License

Choose the license that best fits your needs (for example, MIT or Apache-2.0) and update this section accordingly.