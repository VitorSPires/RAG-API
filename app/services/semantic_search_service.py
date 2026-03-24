"""
Semantic search logic shared by HTTP routes and the LangGraph agent tool.
"""

import logging
from typing import Any, Dict, List

from sqlalchemy import text

from app.core.database import engine
from app.services.embedding_manager import embedding_manager

logger = logging.getLogger(__name__)


def semantic_search_full(
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Full search response with scores and neighbor chunks (same as GET /search)."""
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

        results: List[Dict[str, Any]] = []
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


def semantic_search_for_llm(query: str, limit: int = 5) -> Dict[str, Any]:
    """Lightweight search for LLM consumption (same as GET /search/llm)."""
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
                tc.document_id,
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

        results: List[Dict[str, Any]] = []
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
                "document_id": int(row.document_id),
                "file_name": row.file_name,
                "file_url": (row.file_url or "") if row.file_url is not None else "",
            })

        return {
            "query": query,
            "sources": results
        }
