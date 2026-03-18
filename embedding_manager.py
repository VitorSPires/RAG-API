"""
Module for managing OpenAI embeddings.
"""

import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        load_dotenv()
        
        # Configure OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client configured successfully")
        except Exception as e:
            logger.error(f"Error configuring OpenAI client: {e}")
            raise
        self.model = "text-embedding-3-small"
        self.max_tokens_per_request = 8191  # Model limit
        self.safety_margin = 1000  # Safety margin
        
        # Configure database
        database_url = os.getenv("DATABASE_PUBLIC_URL")
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        self.engine = create_engine(database_url)
    
    def create_embeddings_batch(self, chunks: List[Dict]) -> List[Dict]:
        """
        Create embeddings for a list of chunks in a single batch.

        Args:
            chunks: List of chunks with 'id', 'content', 'overlap_content'.

        Returns:
            List of chunk metadata with embeddings added.
        """
        try:
            # Prepare texts for embeddings (content + overlap)
            texts_for_embedding = []
            chunk_mapping = []
            
            for chunk in chunks:
                # Combine main content with overlap content
                full_text = chunk['content']
                if chunk.get('overlap_content'):
                    full_text = f"{chunk['overlap_content']} {chunk['content']}"
                
                texts_for_embedding.append(full_text)
                chunk_mapping.append(chunk['id'])
            
            logger.info(f"Creating embeddings for {len(texts_for_embedding)} chunks")
            
            # Create embeddings in batch
            response = self.client.embeddings.create(
                input=texts_for_embedding,
                model=self.model
            )
            
            # Process response
            embeddings = []
            for i, embedding_data in enumerate(response.data):
                chunk_id = chunk_mapping[i]
                embedding_vector = embedding_data.embedding
                
                embeddings.append({
                    'chunk_id': chunk_id,
                    'embedding': embedding_vector,
                    'token_count': embedding_data.usage.prompt_tokens if hasattr(embedding_data, 'usage') else None
                })
            
            logger.info(f"Embeddings created successfully: {len(embeddings)} vectors")
            logger.info(f"Tokens used: {response.usage.total_tokens}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def save_embeddings_to_database(self, embeddings: List[Dict]) -> bool:
        """
        Persist embeddings into the database.

        Args:
            embeddings: List of embeddings with 'chunk_id' and 'embedding'.

        Returns:
            True if everything was saved successfully.
        """
        try:
            with self.engine.connect() as connection:
                for embedding_data in embeddings:
                    # Convert embedding into PostgreSQL vector format
                    embedding_vector = embedding_data['embedding']
                    vector_str = f"[{','.join(map(str, embedding_vector))}]"
                    
                    # Insert into embeddings table using direct string formatting
                    sql = f"""
                        INSERT INTO embeddings (chunk_id, embedding_vector, model_name, created_at)
                        VALUES ('{embedding_data['chunk_id']}', '{vector_str}'::vector, '{self.model}', NOW())
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            embedding_vector = EXCLUDED.embedding_vector,
                            model_name = EXCLUDED.model_name,
                            updated_at = NOW()
                    """
                    connection.execute(text(sql))
                
                connection.commit()
                logger.info(f"Embeddings saved to database: {len(embeddings)} vectors")
                return True
                
        except Exception as e:
            logger.error(f"Error saving embeddings to database: {e}")
            return False
    
    def process_document_chunks(self, document_id: int) -> bool:
        """
        Process all chunks of a document to create embeddings.

        Args:
            document_id: Document identifier.

        Returns:
            True if processing completed successfully.
        """
        try:
            # Fetch document chunks
            chunks = self._get_document_chunks(document_id)
            
            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return True
            
            logger.info(f"Processing embeddings for document {document_id}: {len(chunks)} chunks")
            
            # Create embeddings
            embeddings = self.create_embeddings_batch(chunks)
            
            if not embeddings:
                raise Exception(f"No embeddings were created for document {document_id}")
            
            # Save to database
            success = self.save_embeddings_to_database(embeddings)
            
            if not success:
                raise Exception(f"Failed to save embeddings for document {document_id}")
            
            logger.info(f"Embeddings successfully processed for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing embeddings for document {document_id}: {e}")
            return False
    
    def _get_document_chunks(self, document_id: int) -> List[Dict]:
        """
        Fetch chunks for a document from the database.

        Args:
            document_id: Document identifier.

        Returns:
            List of chunk dicts.
        """
        try:
            with self.engine.connect() as connection:
                # First, check if there are chunks for this document
                count_result = connection.execute(text("""
                    SELECT COUNT(*) as count
                    FROM text_chunks 
                    WHERE document_id = :document_id
                """), {"document_id": document_id})
                
                count = count_result.scalar()
                logger.info(f"Found {count} chunks for document {document_id}")
                
                if count == 0:
                    return []
                
                result = connection.execute(text("""
                    SELECT 
                        chunk_id, content, overlap_content
                    FROM text_chunks 
                    WHERE document_id = :document_id
                    ORDER BY chunk_index
                """), {"document_id": document_id})
                
                chunks = []
                for row in result.fetchall():
                    chunks.append({
                        'id': row.chunk_id,
                        'content': row.content,
                        'overlap_content': row.overlap_content or ""
                    })
                
                logger.info(f"Returning {len(chunks)} chunks for embedding")
                return chunks
                
        except Exception as e:
            logger.error(f"Error fetching chunks for document {document_id}: {e}")
            return []
    
    def get_embedding_stats(self) -> Dict:
        """
        Get aggregate statistics about embeddings.

        Returns:
            Dictionary with statistics.
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("""
                    SELECT 
                        COUNT(*) as total_embeddings,
                        COUNT(DISTINCT chunk_id) as unique_chunks,
                        model_name
                    FROM embeddings
                    GROUP BY model_name
                """))
                
                stats = {
                    'total_embeddings': 0,
                    'unique_chunks': 0,
                    'model_name': self.model
                }
                
                row = result.fetchone()
                if row:
                    stats.update({
                        'total_embeddings': row.total_embeddings or 0,
                        'unique_chunks': row.unique_chunks or 0,
                        'model_name': row.model_name or self.model
                    })
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting embeddings stats: {e}")
            return {
                'total_embeddings': 0,
                'unique_chunks': 0,
                'model_name': self.model
            }

# Global instance
embedding_manager = EmbeddingManager() 