"""
FastAPI application entry point.
"""
import logging
from fastapi import FastAPI

from app.core.config import settings
from app.core.database import Base, engine # Import models to register them
from app.api.twilio_routes import router as twilio_router  # Import Twilio routes
from app.models import Agent, Tool  # Import models to register them
from app.api.agent import router as agent_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agent Workspace Platform",
    description="Multi-agent workspace with voice capabilities",
    version="0.1.0",
    debug=settings.debug,
)

# Register routes
app.include_router(twilio_router)
# Register routers
app.include_router(agent_router)


@app.on_event("startup")
async def startup_event():
    """Create database tables automatically on startup."""
    logger.info("Starting up application...")
    # Auto-create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Application startup complete")


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

