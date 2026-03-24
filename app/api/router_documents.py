import logging
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile, Body

from app.services.document_manager import document_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


@router.get("/documents")
async def list_documents():
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
        logger.error("Error listing documents: %s", e)
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/documents/stats")
async def get_documents_stats():
    try:
        stats = document_manager.get_documents_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error("Error getting documents stats: %s", e)
        raise HTTPException(status_code=500, detail=f"Error getting document stats: {str(e)}")


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: int):
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
        logger.error("Error getting chunks for document %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/documents/{document_id}")
async def get_document(document_id: int):
    try:
        document = document_manager.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "success", "data": document}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving document %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
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
        logger.error("Error during upload: %s", e)
        raise HTTPException(status_code=500, detail=f"Error during upload: {str(e)}")


@router.put("/documents/{document_id}")
async def update_document(document_id: int, updates: Dict[str, Any] = Body(...)):
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
        logger.error("Error updating document %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int):
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
        logger.error("Error deleting document %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
