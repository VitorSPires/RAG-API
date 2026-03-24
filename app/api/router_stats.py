import logging

from fastapi import APIRouter, HTTPException

from app.services.document_manager import document_manager
from app.services.embedding_manager import embedding_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stats"])


@router.get("/chunks/stats")
async def get_chunks_stats():
    try:
        stats = document_manager.get_chunks_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error("Error getting chunks stats: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/embeddings/stats")
async def get_embeddings_stats():
    try:
        stats = embedding_manager.get_embedding_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error("Error getting embeddings stats: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
