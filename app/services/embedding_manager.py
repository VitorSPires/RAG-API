"""
Module for managing OpenAI embeddings.
"""

import os
import logging
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import text

from app.core.config import get_embedding_model
from app.core.database import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        try:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client configured successfully")
        except Exception as e:
            logger.error("Error configuring OpenAI client: %s", e)
            raise
        self.model = get_embedding_model()
        self.max_tokens_per_request = 8191
        self.safety_margin = 1000
        self.engine = engine

    def create_embeddings_batch(self, chunks: List[Dict]) -> List[Dict]:
        try:
            texts_for_embedding = []
            chunk_mapping = []

            for chunk in chunks:
                full_text = chunk['content']
                if chunk.get('overlap_content'):
                    full_text = f"{chunk['overlap_content']} {chunk['content']}"

                texts_for_embedding.append(full_text)
                chunk_mapping.append(chunk['id'])

            logger.info("Creating embeddings for %s chunks", len(texts_for_embedding))

            response = self.client.embeddings.create(
                input=texts_for_embedding,
                model=self.model
            )

            embeddings = []
            for i, embedding_data in enumerate(response.data):
                chunk_id = chunk_mapping[i]
                embedding_vector = embedding_data.embedding
                embeddings.append({
                    'chunk_id': chunk_id,
                    'embedding': embedding_vector,
                    'token_count': embedding_data.usage.prompt_tokens if hasattr(embedding_data, 'usage') else None
                })

            logger.info("Embeddings created successfully: %s vectors", len(embeddings))
            logger.info("Tokens used: %s", response.usage.total_tokens)
            return embeddings

        except Exception as e:
            logger.error("Error creating embeddings: %s", e)
            raise

    def save_embeddings_to_database(self, embeddings: List[Dict]) -> bool:
        try:
            with self.engine.connect() as connection:
                for embedding_data in embeddings:
                    embedding_vector = embedding_data['embedding']
                    vector_str = f"[{','.join(map(str, embedding_vector))}]"

                    sql = text("""
                        INSERT INTO embeddings (chunk_id, embedding_vector, model_name, created_at)
                        VALUES (:chunk_id, CAST(:vector_str AS vector), :model_name, NOW())
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            embedding_vector = EXCLUDED.embedding_vector,
                            model_name = EXCLUDED.model_name,
                            updated_at = NOW()
                    """)
                    connection.execute(sql, {
                        "chunk_id": embedding_data["chunk_id"],
                        "vector_str": vector_str,
                        "model_name": self.model,
                    })

                connection.commit()
                logger.info("Embeddings saved to database: %s vectors", len(embeddings))
                return True

        except Exception as e:
            logger.error("Error saving embeddings to database: %s", e)
            return False

    def process_document_chunks(self, document_id: int) -> bool:
        try:
            chunks = self._get_document_chunks(document_id)

            if not chunks:
                logger.warning("No chunks found for document %s", document_id)
                return True

            logger.info("Processing embeddings for document %s: %s chunks", document_id, len(chunks))

            embeddings = self.create_embeddings_batch(chunks)

            if not embeddings:
                raise RuntimeError(f"No embeddings were created for document {document_id}")

            success = self.save_embeddings_to_database(embeddings)

            if not success:
                raise RuntimeError(f"Failed to save embeddings for document {document_id}")

            logger.info("Embeddings successfully processed for document %s", document_id)
            return True

        except Exception as e:
            logger.error("Error processing embeddings for document %s: %s", document_id, e)
            return False

    def _get_document_chunks(self, document_id: int) -> List[Dict]:
        try:
            with self.engine.connect() as connection:
                count_result = connection.execute(text("""
                    SELECT COUNT(*) as count
                    FROM text_chunks
                    WHERE document_id = :document_id
                """), {"document_id": document_id})

                count = count_result.scalar()
                logger.info("Found %s chunks for document %s", count, document_id)

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

                logger.info("Returning %s chunks for embedding", len(chunks))
                return chunks

        except Exception as e:
            logger.error("Error fetching chunks for document %s: %s", document_id, e)
            return []

    def get_embedding_stats(self) -> Dict:
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
            logger.error("Error getting embeddings stats: %s", e)
            return {
                'total_embeddings': 0,
                'unique_chunks': 0,
                'model_name': self.model
            }


embedding_manager = EmbeddingManager()
