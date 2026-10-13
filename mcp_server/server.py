"""MCP server exposing document-search and task tools to an AI client."""
from mcp.server.fastmcp import FastMCP
from app.db import SessionLocal, initialize_database
from app.models import Document
from app.study import generate_study_plan as generate_plan, list_study_tasks as get_study_tasks, update_study_task as set_study_status
from app.rag import search_documents

mcp = FastMCP("knowledge-action-tools")


@mcp.tool()
def list_documents() -> list[dict]:
    """List the PDF documents currently stored in the knowledge base."""
    initialize_database()
    with SessionLocal() as db:
        documents = db.query(Document).order_by(Document.id.desc()).all()
        return [
            {"id": item.id, "filename": item.filename, "created_at": item.created_at.isoformat() if item.created_at else None}
            for item in documents
        ]


@mcp.tool()
def search_documents_tool(query: str, top_k: int = 4) -> list[dict]:
    """Search uploaded document chunks by semantic similarity. Use for questions about the knowledge base."""
    initialize_database()
    with SessionLocal() as db:
        return search_documents(db, query, top_k=max(1, min(top_k, 8)))


@mcp.tool()
async def generate_study_plan(document_id: int, task_count: int = 5) -> dict:
    """Generate and SAVE a study checklist grounded in an uploaded PDF. Only call if the user requests saving a plan."""
    return await generate_plan(document_id, task_count)


@mcp.tool()
def list_study_tasks(document_id: int | None = None) -> list[dict]:
    """List saved study tasks, optionally filtered by document ID."""
    return get_study_tasks(document_id)


@mcp.tool()
def update_study_task_status(task_id: int, status: str) -> dict:
    """Set a study task status to pending, in_progress, or completed. Only update when requested."""
    return set_study_status(task_id, status)


if __name__ == "__main__":
    initialize_database()
    mcp.run(transport="stdio")
