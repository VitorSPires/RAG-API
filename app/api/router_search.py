import logging

from fastapi import APIRouter, HTTPException, Query

from app.services.semantic_search_service import semantic_search_for_llm, semantic_search_full

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.get("/search")
async def semantic_search(
    query: str = Query(..., description="Text for semantic search"),
    limit: int = Query(5, description="Maximum number of results", ge=1, le=20),
    similarity_threshold: float = Query(0.0, description="Similarity threshold (0.0 to 1.0)", ge=0.0, le=1.0)
):
    try:
        return semantic_search_full(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )
    except Exception as e:
        logger.error("Error during semantic search: %s", e)
        raise HTTPException(status_code=500, detail=f"Error during semantic search: {str(e)}")


@router.get("/search/llm")
async def semantic_search_llm(
    query: str = Query(..., description="Text for semantic search"),
    limit: int = Query(5, description="Maximum number of results", ge=1, le=20)
):
    try:
        return semantic_search_for_llm(query=query, limit=limit)
    except Exception as e:
        logger.error("Error during semantic search for LLM: %s", e)
        raise HTTPException(status_code=500, detail=f"Error during semantic search for LLM: {str(e)}")
