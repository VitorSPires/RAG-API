-- Changeset: 001_install_pgvector
-- Description: Install pgvector extension in the database
-- Date: 2025-07-08

-- Install pgvector extension (if it does not exist)
CREATE EXTENSION IF NOT EXISTS vector; 

-- Verify that the extension was installed correctly
SELECT * FROM pg_extension WHERE extname = 'vector';