import logging
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logger = logging.getLogger(__name__)


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql://", 1)
    return url


def _can_connect(database_url: str) -> bool:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database connection test failed for URL '%s': %s", database_url, exc)
        return False
    finally:
        engine.dispose()


@lru_cache(maxsize=1)
def resolve_database_url() -> str:
    """
    Resolve a working database URL.

    Resolution order:
    1) DATABASE_URL (internal/private)
    2) DATABASE_PUBLIC_URL (public fallback)
    """
    internal_url = (os.getenv("DATABASE_URL") or "").strip()
    public_url = (os.getenv("DATABASE_PUBLIC_URL") or "").strip()

    if internal_url:
        normalized_internal = _normalize_database_url(internal_url)
        if _can_connect(normalized_internal):
            logger.info("Using internal DATABASE_URL for PostgreSQL connection")
            return normalized_internal
        logger.warning("Internal DATABASE_URL is set but unreachable. Falling back to DATABASE_PUBLIC_URL")

    if public_url:
        normalized_public = _normalize_database_url(public_url)
        if _can_connect(normalized_public):
            logger.info("Using DATABASE_PUBLIC_URL for PostgreSQL connection")
            return normalized_public
        raise ValueError("DATABASE_PUBLIC_URL is set but unreachable")

    raise ValueError("No valid database URL found. Set DATABASE_URL or DATABASE_PUBLIC_URL")


def create_database_engine():
    database_url = resolve_database_url()
    return create_engine(database_url, pool_pre_ping=True)


def get_embedding_model() -> str:
    return (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()


def get_chat_model() -> str:
    return (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()


def get_optional_agent_system_prompt() -> Optional[str]:
    """
    Optional extra system message for the ReAct loop only.
    Behavior for replies and sources is driven mainly by AgentAnswerPayload field descriptions.
    """
    custom = (os.getenv("AGENT_SYSTEM_PROMPT") or "").strip()
    return custom or None


def get_cors_allowed_origins():
    raw = (os.getenv("CORS_ALLOWED_ORIGINS") or "").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
