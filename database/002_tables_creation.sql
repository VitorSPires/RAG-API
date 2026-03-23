-- Script to drop and recreate tables with the correct structure

-- 1. Drop existing tables (if they exist)
DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS text_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- 2. Create table for processed documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(128) NOT NULL UNIQUE,         -- Stable file identifier
    file_name TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,               -- SHA256 hash of file content
    file_url TEXT NOT NULL,                       -- Public URL to access the file
    processed_at TIMESTAMPTZ DEFAULT now()
);

-- 3. Create table for text chunks
CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,                    -- Main text chunk (up to ~250 tokens)
    overlap_content TEXT DEFAULT '',          -- Overlap region (up to ~50 tokens, when needed)
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    start_position INTEGER NOT NULL,
    end_position INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create table for embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) NOT NULL REFERENCES text_chunks(chunk_id) ON DELETE CASCADE,
    embedding_vector vector(1536),  -- embedding dimension for text-embedding-3-small
    model_name VARCHAR(100) DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chunk_id)
);

-- 5. Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON text_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON text_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_token_count ON text_chunks(token_count);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
