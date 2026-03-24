import logging

from sqlalchemy import text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import create_database_engine

logger = logging.getLogger(__name__)

engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def test_database_connection() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return True
    except Exception as e:
        logger.error("Error connecting to database: %s", e)
        return False


def setup_pgvector() -> None:
    try:
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            connection.commit()
            logger.info("pgvector extension configured successfully")
    except Exception as e:
        logger.warning("Warning: could not configure pgvector: %s", e)
