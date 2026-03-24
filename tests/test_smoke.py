import os

import pytest


def test_normalize_database_url():
    from app.core.config import _normalize_database_url

    assert _normalize_database_url("postgres://u:p@h/db").startswith("postgresql://")
    assert _normalize_database_url("postgresql://x") == "postgresql://x"


@pytest.mark.integration
def test_health_when_configured():
    """Loads the full app; requires .env with DB + OpenAI (same as running the API)."""
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if not (os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL")):
        pytest.skip("No database URL configured")

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert "database" in body
