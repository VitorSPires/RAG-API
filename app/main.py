import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.agent.graph import build_agent_graph
from app.agent.schemas import AgentAnswerPayload
from app.api.router_chat import router as chat_router
from app.api.router_documents import router as documents_router
from app.api.router_health import router as health_router
from app.api.router_search import router as search_router
from app.api.router_stats import router as stats_router
from app.api.router_ui import router as ui_router
from app.core.config import get_cors_allowed_origins, resolve_database_url
from app.core.database import setup_pgvector, test_database_connection

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application")

    app.state.checkpointer = None
    app.state.psycopg_pool = None
    app.state.agent_graph = None

    pool: ConnectionPool | None = None

    if not test_database_connection():
        logger.error("Failed to connect to database - check credentials and connectivity")
        logger.warning("Application started without a working database connection")
    else:
        setup_pgvector()
        try:
            db_url = resolve_database_url()
            pool = ConnectionPool(
                conninfo=db_url,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            )
            # Register app types so checkpoint serde does not warn (and future strict mode works).
            # LangGraph always allows SAFE_MSGPACK_TYPES (incl. ToolMessage); this adds our schema.
            checkpoint_serde = JsonPlusSerializer(allowed_msgpack_modules=[AgentAnswerPayload])
            checkpointer = PostgresSaver(pool, serde=checkpoint_serde)
            checkpointer.setup()
            app.state.checkpointer = checkpointer
            app.state.psycopg_pool = pool
            app.state.agent_graph = build_agent_graph(checkpointer)
            logger.info("LangGraph Postgres checkpointer initialized")
        except Exception as e:
            logger.warning("LangGraph checkpointer not initialized: %s", e)
            if pool is not None:
                pool.close()
                pool = None

    yield

    logger.info("Shutting down FastAPI application")
    p = getattr(app.state, "psycopg_pool", None)
    if p is not None:
        p.close()


app = FastAPI(
    title="Vector Database API",
    description="API for RAG with PostgreSQL + pgvector, semantic search, and LangGraph agent.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ui_router)
app.include_router(documents_router)
app.include_router(stats_router)
app.include_router(search_router)
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
