from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_PATH = REPO_ROOT / "templates" / "documents.html"


@router.get("/", response_class=HTMLResponse)
async def root():
    try:
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Vector Database API</title></head>
            <body>
                <h1>Vector Database API</h1>
                <p>Interface not found. Make sure templates/documents.html exists.</p>
                <p><a href="/documents">View documents (JSON)</a></p>
            </body>
        </html>
        """)


@router.get("/interface", response_class=HTMLResponse)
async def documents_interface():
    try:
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Documents Interface</title></head>
            <body>
                <h1>Documents Interface</h1>
                <p>File templates/documents.html not found.</p>
                <p><a href="/documents">View documents (JSON)</a></p>
            </body>
        </html>
        """)
