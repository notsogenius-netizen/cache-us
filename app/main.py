"""
FastAPI application entry point.
"""
import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

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

# Add CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to log WebSocket connection attempts
class WebSocketLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log WebSocket upgrade requests
        if request.url.path.startswith("/twilio/ws"):
            logger.info(f"[Middleware] WebSocket connection attempt to: {request.url}")
            print(f"[Middleware] ====== WebSocket connection attempt ======")
            print(f"[Middleware] Path: {request.url.path}")
            print(f"[Middleware] Query: {request.url.query}")
            print(f"[Middleware] Headers: {dict(request.headers)}")
        return await call_next(request)

app.add_middleware(WebSocketLoggingMiddleware)

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


@app.websocket("/test-ws")
async def test_websocket(websocket: WebSocket):
    """Test WebSocket endpoint to verify WebSocket functionality."""
    logger.info("[Test] WebSocket test endpoint hit!")
    print("[Test] ====== WebSocket test endpoint hit! ======")
    await websocket.accept()
    try:
        await websocket.send_json({"message": "WebSocket connection successful!"})
        while True:
            data = await websocket.receive_text()
            logger.info(f"[Test] Received: {data}")
            await websocket.send_json({"echo": data})
    except Exception as e:
        logger.error(f"[Test] WebSocket error: {e}")
        print(f"[Test] WebSocket error: {e}")

