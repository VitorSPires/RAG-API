"""
Module for managing documents and their lifecycle.
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from document_processor import document_processor, TextChunk
from firebase_storage import firebase_storage
from embedding_manager import embedding_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self):
        load_dotenv()
        self.database_url = os.getenv("DATABASE_PUBLIC_URL")
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        
        self.engine = create_engine(self.database_url)
    
    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash for the given file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def get_all_documents(self) -> List[Dict]:
        """Retrieve all documents from the database."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        id,
                        file_id,
                        file_name,
                        file_hash,
                        file_url,
                        processed_at
                    FROM documents 
                    ORDER BY processed_at DESC
                """))
                
                documents = []
                for row in result.fetchall():
                    documents.append({
                        'id': row.id,
                        'file_id': row.file_id,
                        'file_name': row.file_name,
                        'file_hash': row.file_hash,
                        'file_url': row.file_url,
                        'processed_at': row.processed_at.isoformat() if row.processed_at else None
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """Retrieve a single document by its ID."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        id,
                        file_id,
                        file_name,
                        file_hash,
                        file_url,
                        processed_at
                    FROM documents 
                    WHERE id = :document_id
                """), {"document_id": document_id})
                
                row = result.fetchone()
                if row:
                    return {
                        'id': row.id,
                        'file_id': row.file_id,
                        'file_name': row.file_name,
                        'file_hash': row.file_hash,
                        'file_url': row.file_url,
                        'processed_at': row.processed_at.isoformat() if row.processed_at else None
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error fetching document {document_id}: {e}")
            return None
    
    def add_document(self, file_name: str, file_content: bytes, file_type: str = None) -> Optional[Dict]:
        """Add a new document and process it into chunks and embeddings."""
        try:
            file_hash = self._calculate_file_hash(file_content)
            file_id = f"local_{file_hash[:16]}"  # Local ID derived from hash
            
            with self.engine.connect() as connection:
                # Check if file already exists
                existing = connection.execute(text("""
                    SELECT id FROM documents WHERE file_hash = :file_hash
                """), {"file_hash": file_hash}).fetchone()
                
                if existing:
                    return {
                        "error": "File already exists in the database",
                        "document_id": existing.id
                    }
                
                # Insert a new document first to obtain its ID
                result = connection.execute(text("""
                    INSERT INTO documents (file_id, file_name, file_hash, file_url)
                    VALUES (:file_id, :file_name, :file_hash, '')
                    RETURNING id
                """), {
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_hash": file_hash
                })
                
                document_id = result.scalar()
                
                # Upload to Firebase Storage using the correct ID (optional)
                firebase_url = firebase_storage.upload_file(file_content, file_name, document_id)
                if firebase_url:
                    connection.execute(text("""
                        UPDATE documents SET file_url = :file_url WHERE id = :document_id
                    """), {
                        "file_url": firebase_url,
                        "document_id": document_id
                    })
                    logger.info(f"Document saved with ID: {document_id} and uploaded to Firebase")
                else:
                    logger.info(
                        "Document saved with ID: %s (Firebase Storage not configured or upload skipped)",
                        document_id,
                    )
                
                # Process document into chunks
                try:
                    chunks = document_processor.process_document(file_content, file_name, document_id)
                    
                    if chunks:
                        # Persist chunks
                        self._save_chunks(chunks, connection)
                        logger.info(f"Document processed: {len(chunks)} chunks created")
                        
                        # Commit chunks before creating embeddings
                        connection.commit()
                        
                        # Create embeddings for chunks
                        logger.info(f"Starting embeddings creation for document {document_id}")
                        embedding_manager.process_document_chunks(document_id)
                        logger.info(f"Embeddings successfully created for document {document_id}")
                    else:
                        logger.warning(f"No chunks were created for document {file_name}")
                        
                except Exception as e:
                    logger.error(f"Error while processing chunks: {e}")
                    connection.rollback()
                    raise
                
                # Final commit
                connection.commit()
                
                logger.info(f"Document added: {file_name} (ID: {document_id})")
                
                return self.get_document_by_id(document_id)
                
        except Exception as e:
            logger.error(f"Error while adding document: {e}")
            return None
    
    def update_document(self, document_id: int, updates: Dict) -> Optional[Dict]:
        """Update document metadata."""
        try:
            # Allowed fields for update
            allowed_fields = ['file_name', 'file_url']
            update_fields = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not update_fields:
                return {"error": "No valid fields to update"}
            
            # Build dynamic UPDATE query
            set_clause = ", ".join([f"{field} = :{field}" for field in update_fields.keys()])
            
            query = f"""
                UPDATE documents 
                SET {set_clause}
                WHERE id = :document_id
                RETURNING id
            """
            
            with self.engine.connect() as connection:
                result = connection.execute(text(query), {
                    **update_fields,
                    "document_id": document_id
                })
                
                if result.rowcount == 0:
                    return {"error": "Document not found"}
                
                connection.commit()
                logger.info(f"Document updated: ID {document_id}")
                
                return self.get_document_by_id(document_id)
                
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return None
    
    def delete_document(self, document_id: int) -> Dict:
        """Delete a document and related data."""
        try:
            with self.engine.connect() as connection:
                # Check if document exists
                existing = connection.execute(text("""
                    SELECT file_name, file_url FROM documents WHERE id = :document_id
                """), {"document_id": document_id}).fetchone()
                
                if not existing:
                    return {"error": "Document not found"}
                
                # Delete related chunks first
                connection.execute(text("""
                    DELETE FROM text_chunks WHERE document_id = :document_id
                """), {"document_id": document_id})
                
                # Delete file from Firebase Storage
                if existing.file_url and existing.file_url.startswith('https://'):
                    firebase_storage.delete_file(existing.file_url)
                
                # Delete document record
                result = connection.execute(text("""
                    DELETE FROM documents WHERE id = :document_id
                """), {"document_id": document_id})
                
                connection.commit()
                
                logger.info(f"Document deleted: {existing.file_name} (ID: {document_id})")
                
                return {
                    "success": True,
                    "message": f"Document '{existing.file_name}' deleted successfully"
                }
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return {"error": f"Error deleting document: {str(e)}"}
    
    def get_documents_stats(self) -> Dict:
        """Get high-level statistics about stored documents."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        COUNT(*) as total_documents,
                        MAX(processed_at) as last_upload
                    FROM documents
                """))
                
                row = result.fetchone()
                
                # Approximate total size (based on text fields)
                size_result = connection.execute(text("""
                    SELECT 
                        COALESCE(SUM(LENGTH(file_name) + LENGTH(file_hash) + LENGTH(file_url)), 0) as total_size
                    FROM documents
                """))
                
                total_size = size_result.scalar() or 0
                
                # Format size as human-readable string
                if total_size > 0:
                    if total_size < 1024:
                        size_str = f"{total_size} B"
                    elif total_size < 1024 * 1024:
                        size_str = f"{total_size / 1024:.1f} KB"
                    else:
                        size_str = f"{total_size / (1024 * 1024):.1f} MB"
                else:
                    size_str = "0 B"
                
                # Format last upload date
                last_upload = "Never"
                if row.last_upload:
                    from datetime import datetime
                    last_upload = row.last_upload.strftime("%Y-%m-%d")
                
                return {
                    'total_documents': row.total_documents or 0,
                    'total_size': size_str,
                    'last_upload': last_upload
                }
                
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {
                'total_documents': 0,
                'total_size': '0 B',
                'last_upload': 'Never'
            }
    
    def _save_chunks(self, chunks: List[TextChunk], connection) -> None:
        """Persist chunks into the database."""
        try:
            logger.info(f"Saving {len(chunks)} chunks into the database")
            
            for i, chunk in enumerate(chunks):
                connection.execute(text("""
                    INSERT INTO text_chunks (
                        chunk_id, document_id, content, overlap_content,
                        chunk_index, token_count, start_position, end_position
                    ) VALUES (
                        :chunk_id, :document_id, :content, :overlap_content,
                        :chunk_index, :token_count, :start_position, :end_position
                    )
                """), {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "overlap_content": chunk.overlap_content,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "start_position": chunk.start_position,
                    "end_position": chunk.end_position
                })
                
                if i < 3:  # Log only the first 3 chunks for debugging
                    logger.info(f"Chunk {i+1} saved: {chunk.id} (doc_id: {chunk.document_id})")
            
            logger.info(f"All {len(chunks)} chunks were saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            raise
    
    def get_document_chunks(self, document_id: int) -> List[Dict]:
        """Get all chunks belonging to a specific document."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        id, chunk_id, content, overlap_content,
                        chunk_index, token_count, start_position, end_position,
                        created_at
                    FROM text_chunks 
                    WHERE document_id = :document_id
                    ORDER BY chunk_index
                """), {"document_id": document_id})
                
                chunks = []
                for row in result.fetchall():
                    chunks.append({
                        'id': row.id,
                        'chunk_id': row.chunk_id,
                        'content': row.content,
                        'overlap_content': row.overlap_content,
                        'chunk_index': row.chunk_index,
                        'token_count': row.token_count,
                        'start_position': row.start_position,
                        'end_position': row.end_position,
                        'created_at': row.created_at.isoformat() if row.created_at else None
                    })
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error fetching chunks for document {document_id}: {e}")
            return []
    
    def get_chunks_stats(self) -> Dict:
        """Get aggregate statistics about text chunks."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        AVG(token_count) as avg_tokens,
                        MIN(token_count) as min_tokens,
                        MAX(token_count) as max_tokens,
                        SUM(token_count) as total_tokens
                    FROM text_chunks
                """))
                
                row = result.fetchone()
                return {
                    'total_chunks': row.total_chunks or 0,
                    'avg_tokens': round(row.avg_tokens, 1) if row.avg_tokens else 0,
                    'min_tokens': row.min_tokens or 0,
                    'max_tokens': row.max_tokens or 0,
                    'total_tokens': row.total_tokens or 0
                }
                
        except Exception as e:
            logger.error(f"Error getting chunks stats: {e}")
            return {
                'total_chunks': 0,
                'avg_tokens': 0,
                'min_tokens': 0,
                'max_tokens': 0,
                'total_tokens': 0
            }
    
# Global instance
document_manager = DocumentManager() 