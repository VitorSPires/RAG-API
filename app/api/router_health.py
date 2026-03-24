import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import text

from app.core.database import engine, test_database_connection

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint for the API and database."""
    db_status = "connected" if test_database_connection() else "disconnected"
    return {
        "status": "healthy",
        "database": db_status,
        "message": "Application is running"
    }


@router.get("/test-db")
async def test_database():
    """Test database connection and pgvector."""
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
