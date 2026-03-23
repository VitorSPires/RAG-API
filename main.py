from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import declarative_base, sessionmaker
import logging
from document_manager import document_manager
from embedding_manager import embedding_manager
from config import create_database_engine, get_cors_allowed_origins

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy configuration
engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def test_database_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return True
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return False


def setup_pgvector():
    try:
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            connection.commit()
            logger.info("pgvector extension configured successfully")
    except Exception as e:
        logger.warning(f"Warning: could not configure pgvector: {e}")

# Context manager for application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI application")

    if not test_database_connection():
        logger.error("Failed to connect to database - check credentials and connectivity")
        # Do not fail startup, only log the error
        logger.warning("Application started without a working database connection")
    else:
        # Configure pgvector only if connection is working
        setup_pgvector()

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application")

# Create FastAPI application instance
app = FastAPI(
    title="Vector Database API",
    description="API for communicating with a vector database using PostgreSQL + pgvector",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """Health check endpoint for the API and database."""
    db_status = "connected" if test_database_connection() else "disconnected"
    return {
        "status": "healthy",
        "database": db_status,
        "message": "Application is running"
    }

@app.get("/test-db")
async def test_database():
    """Endpoint to test database connection and pgvector functionality."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1 as test"))
            test_value = result.scalar()
            
            version_result = connection.execute(text("SELECT version()"))
            version = version_result.scalar()
            postgres_version = version.split(" ")[1] if version else "version not found"
            
            pgvector_result = connection.execute(text("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                ) as pgvector_installed
            """))
            pgvector_installed = pgvector_result.scalar()
            
            vector_test = None
            if pgvector_installed:
                try:
                    connection.execute(text("""
                        CREATE TEMP TABLE temp_test_vectors (
                            id SERIAL PRIMARY KEY,
                            embedding vector(3)
                        )
                    """))
                    
                    connection.execute(text("""
                        INSERT INTO temp_test_vectors (embedding) 
                        VALUES ('[1,2,3]'::vector)
                    """))
                    
                    vector_result = connection.execute(text("""
                        SELECT embedding FROM temp_test_vectors LIMIT 1
                    """))
                    
                    vector_test = str(vector_result.scalar())
                except Exception as e:
                    vector_test = f"Error: {str(e)}"
            
            return {
                "status": "success",
                "message": "Database connection established successfully",
                "database_info": {
                    "connection_test": test_value,
                    "postgres_version": postgres_version,
                    "pgvector_installed": pgvector_installed,
                    "vector_test": vector_test
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error while connecting to database: {str(e)}",
            "database_info": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/", response_class=HTMLResponse)
async def root():
    """HTML interface for document management."""
    try:
        with open("templates/documents.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Vector Database API</title></head>
            <body>
                <h1>Vector Database API</h1>
                <p>Interface not found. Make sure templates/documents.html exists.</p>
                <p><a href="/documents">View documents (JSON)</a></p>
            </body>
        </html>
        """)

@app.get("/interface", response_class=HTMLResponse)
async def documents_interface():
    """HTML interface for document management."""
    try:
        with open("templates/documents.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Documents Interface</title></head>
            <body>
                <h1>Documents Interface</h1>
                <p>File templates/documents.html not found.</p>
                <p><a href="/documents">View documents (JSON)</a></p>
            </body>
        </html>
        """)

@app.get("/documents")
async def list_documents():
    """List all documents and return management statistics."""
    try:
        documents = document_manager.get_all_documents()
        stats = document_manager.get_documents_stats()
        
        return {
            "status": "success",
            "data": {
                "documents": documents,
                "stats": stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    """Get a specific document by ID."""
    try:
        document = document_manager.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "data": document
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a new document."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name is required")

        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        result = document_manager.add_document(
            file_name=file.filename,
            file_content=file_content,
            file_type=file.content_type
        )
        
        if result and "error" in result:
            return {
                "status": "warning",
                "message": result["error"],
                "data": result
            }
        
        if not result:
            raise HTTPException(status_code=500, detail="Error while adding document")
        
        return {
            "status": "success",
            "message": "Document added successfully",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error during upload: {str(e)}")

@app.put("/documents/{document_id}")
async def update_document(document_id: int, updates: dict):
    """Update document metadata."""
    try:
        result = document_manager.update_document(document_id, updates)
        
        if result and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "message": "Document updated successfully",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document and its related data."""
    try:
        result = document_manager.delete_document(document_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {
            "status": "success",
            "message": result["message"],
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/documents/stats")
async def get_documents_stats():
    """Get high-level statistics about documents."""
    try:
        stats = document_manager.get_documents_stats()
        
        return {
            "status": "success",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting documents stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document stats: {str(e)}")

@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: int):
    """Get chunks of a specific document."""
    try:
        chunks = document_manager.get_document_chunks(document_id)
        return {
            "status": "success",
            "data": {
                "document_id": document_id,
                "chunks": chunks,
                "total_chunks": len(chunks)
            }
        }
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/chunks/stats")
async def get_chunks_stats():
    """Get statistics for all chunks."""
    try:
        stats = document_manager.get_chunks_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting chunks stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/embeddings/stats")
async def get_embeddings_stats():
    """Get statistics for stored embeddings."""
    try:
        stats = embedding_manager.get_embedding_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting embeddings stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/search")
async def semantic_search(
    query: str = Query(..., description="Text for semantic search"),
    limit: int = Query(5, description="Maximum number of results", ge=1, le=20),
    similarity_threshold: float = Query(0.0, description="Similarity threshold (0.0 to 1.0)", ge=0.0, le=1.0)
):
    """
    Perform semantic search using vector embeddings.

    This endpoint converts the query into an embedding and finds the most similar chunks
    in the vector database, optionally returning previous and next chunks as context.

    - **query**: Text to search for.
    - **limit**: Maximum number of results (1–20).
    - **similarity_threshold**: Minimum similarity threshold (0.0–1.0).
    """
    try:
        query_embedding = embedding_manager.client.embeddings.create(
            input=[query],
            model=embedding_manager.model
        )
        
        query_vector = query_embedding.data[0].embedding
        vector_str = f"[{','.join(map(str, query_vector))}]"
        distance_threshold = 1.0 - similarity_threshold
        
        with engine.connect() as connection:
            sql = text("""
                SELECT 
                    tc.chunk_id,
                    tc.content,
                    tc.overlap_content,
                    tc.chunk_index,
                    tc.token_count,
                    d.file_name,
                    d.id as document_id,
                    e.embedding_vector <=> CAST(:vector_str AS vector) as similarity
                FROM text_chunks tc
                JOIN documents d ON tc.document_id = d.id
                JOIN embeddings e ON tc.chunk_id = e.chunk_id
                WHERE e.embedding_vector <=> CAST(:vector_str AS vector) < :distance_threshold
                ORDER BY e.embedding_vector <=> CAST(:vector_str AS vector)
                LIMIT :limit
            """)
            result = connection.execute(sql, {
                "vector_str": vector_str,
                "distance_threshold": distance_threshold,
                "limit": limit,
            })
            
            results = []
            for row in result.fetchall():
                prev_chunk = None
                if row.chunk_index > 0:
                    prev_result = connection.execute(text("""
                        SELECT chunk_id, content, chunk_index
                        FROM text_chunks 
                        WHERE document_id = :doc_id AND chunk_index = :prev_index
                    """), {
                        "doc_id": row.document_id,
                        "prev_index": row.chunk_index - 1
                    })
                    prev_row = prev_result.fetchone()
                    if prev_row:
                        prev_chunk = {
                            "chunk_id": prev_row.chunk_id,
                            "content": prev_row.content,
                            "chunk_index": prev_row.chunk_index
                        }
                
                next_chunk = None
                next_result = connection.execute(text("""
                    SELECT chunk_id, content, chunk_index
                    FROM text_chunks 
                    WHERE document_id = :doc_id AND chunk_index = :next_index
                """), {
                    "doc_id": row.document_id,
                    "next_index": row.chunk_index + 1
                })
                next_row = next_result.fetchone()
                if next_row:
                    next_chunk = {
                        "chunk_id": next_row.chunk_id,
                        "content": next_row.content,
                        "chunk_index": next_row.chunk_index
                    }
                
                results.append({
                    "chunk_id": row.chunk_id,
                    "content": row.content,
                    "overlap_content": row.overlap_content,
                    "chunk_index": row.chunk_index,
                    "token_count": row.token_count,
                    "document": {
                        "id": row.document_id,
                        "file_name": row.file_name
                    },
                    "similarity_score": 1.0 - row.similarity,
                    "similarity_percentage": round((1.0 - row.similarity) * 100, 2),
                    "context": {
                        "previous_chunk": prev_chunk,
                        "next_chunk": next_chunk
                    }
                })
            
            return {
                "status": "success",
                "query": query,
                "total_results": len(results),
                "similarity_threshold": similarity_threshold,
                "results": results
            }

    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        raise HTTPException(status_code=500, detail=f"Error during semantic search: {str(e)}")

@app.get("/search/llm")
async def semantic_search_for_llm(
    query: str = Query(..., description="Text for semantic search"),
    limit: int = Query(5, description="Maximum number of results", ge=1, le=20)
):
    """
    Semantic search endpoint optimized for use with LLMs.

    Returns only the essential data required for a language model to answer a user query:
    - Original query.
    - Relevant texts (chunk plus previous/next context).
    - File name.
    - File URL.

    - **query**: Text to search for.
    - **limit**: Maximum number of results (1–20).
    """
    try:
        query_embedding = embedding_manager.client.embeddings.create(
            input=[query],
            model=embedding_manager.model
        )
        
        query_vector = query_embedding.data[0].embedding
        vector_str = f"[{','.join(map(str, query_vector))}]"
        
        with engine.connect() as connection:
            sql = text("""
                SELECT 
                    tc.chunk_id,
                    tc.content,
                    tc.chunk_index,
                    d.file_name,
                    d.file_url,
                    e.embedding_vector <=> CAST(:vector_str AS vector) as similarity
                FROM text_chunks tc
                JOIN documents d ON tc.document_id = d.id
                JOIN embeddings e ON tc.chunk_id = e.chunk_id
                ORDER BY e.embedding_vector <=> CAST(:vector_str AS vector)
                LIMIT :limit
            """)
            result = connection.execute(sql, {
                "vector_str": vector_str,
                "limit": limit,
            })
            
            results = []
            for row in result.fetchall():
                prev_content = ""
                if row.chunk_index > 0:
                    prev_result = connection.execute(text("""
                        SELECT content
                        FROM text_chunks 
                        WHERE document_id = (SELECT document_id FROM text_chunks WHERE chunk_id = :chunk_id) 
                        AND chunk_index = :prev_index
                    """), {
                        "chunk_id": row.chunk_id,
                        "prev_index": row.chunk_index - 1
                    })
                    prev_row = prev_result.fetchone()
                    if prev_row:
                        prev_content = prev_row.content + " "
                
                next_content = ""
                next_result = connection.execute(text("""
                    SELECT content
                    FROM text_chunks 
                    WHERE document_id = (SELECT document_id FROM text_chunks WHERE chunk_id = :chunk_id) 
                    AND chunk_index = :next_index
                """), {
                    "chunk_id": row.chunk_id,
                    "next_index": row.chunk_index + 1
                })
                next_row = next_result.fetchone()
                if next_row:
                    next_content = " " + next_row.content
                
                full_text = prev_content + row.content + next_content
                
                results.append({
                    "text": full_text.strip(),
                    "file_name": row.file_name,
                    "file_url": row.file_url
                })
            
            return {
                "query": query,
                "sources": results
            }

    except Exception as e:
        logger.error(f"Error during semantic search for LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Error during semantic search for LLM: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    # Use environment variable or localhost default
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)