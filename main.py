"""
Backward-compatible entry point.

Prefer: uvicorn app.main:app
"""
from app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
