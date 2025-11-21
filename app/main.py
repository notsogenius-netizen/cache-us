"""
FastAPI application entry point.
"""
from fastapi import FastAPI

from app.core.config import settings
from app.core.database import Base, engine
from app.models import Agent, Tool, AgentTool  # Import models to register them


# Initialize FastAPI app
app = FastAPI(
    title="Agent Workspace Platform",
    description="Multi-agent workspace with voice capabilities",
    version="0.1.0",
    debug=settings.debug,
)


@app.on_event("startup")
async def startup_event():
    """Create database tables automatically on startup."""
    # Auto-create all tables
    Base.metadata.create_all(bind=engine)


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "message": "Agent Workspace Platform API",
        "version": "0.1.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

