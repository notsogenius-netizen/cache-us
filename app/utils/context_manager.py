"""
Context manager for caching conversation context per phone number.
"""
from typing import Dict, Optional, List, Any
import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from urllib.parse import quote, unquote

from app.models.agent import Agent
from app.models.tool import Tool
from app.core.database import SessionLocal
from app.core.config import settings

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class ConversationContext:
    """Represents conversation context for a phone number."""

    def __init__(
        self,
        phone_number: str,
        agent: Agent,
        tools: List[Tool],
        websocket: Any,  # WebSocket connection
    ):
        self.phone_number = phone_number
        self.agent = agent
        self.namespace = agent.namespace
        self.prompt = agent.prompt
        self.tools = tools
        self.websocket = websocket
        self.conversation_history: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.stream_sid: Optional[str] = None  # Stream SID for sending audio back

    def add_user_message(self, text: str):
        """Add user message to conversation history."""
        self.conversation_history.append({"role": "user", "content": text})
        self.last_activity = datetime.now()

    def add_assistant_message(self, text: str):
        """Add assistant message to conversation history."""
        self.conversation_history.append({"role": "assistant", "content": text})
        self.last_activity = datetime.now()

    def is_expired(self, timeout_minutes: int = 5) -> bool:
        """Check if context has expired (no activity for timeout period)."""
        if self.websocket is not None:
            return False  # Active connection, not expired
        elapsed = datetime.now() - self.last_activity
        return elapsed > timedelta(minutes=timeout_minutes)


class ContextManager:
    """Manages conversation contexts per phone number."""

    def __init__(self):
        self._contexts: Dict[str, ConversationContext] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60  # Check every minute

    async def get_or_create_context(
        self, phone_number: Optional[str], websocket: Any
    ) -> Optional[ConversationContext]:
        """
        Get existing context or create new one for phone number.

        Args:
            phone_number: Phone number (optional, if None will use first available agent)
            websocket: WebSocket connection

        Returns:
            ConversationContext or None if agent not found
        """
        # Use a default key if no phone number provided
        context_key = phone_number or "default"
        
        # Check if context exists
        if context_key in self._contexts:
            context = self._contexts[context_key]
            context.websocket = websocket  # Update websocket
            context.last_activity = datetime.now()
            return context

        # Create new context - fetch agent from database
        db = SessionLocal()
        try:
            agent = None
            
            # If phone_number is provided, try to fetch by phone number first
            if phone_number:
                # Fetch agent by phone number
                # Try both formats: decoded (+17245422869) and URL-encoded (%2B17245422869)
                # This handles cases where the database might have stored the URL-encoded version
                agent = db.query(Agent).filter(Agent.phone_number == phone_number).first()
                
                # If not found, try URL-encoded version
                if not agent:
                    url_encoded_phone = quote(phone_number, safe='')
                    agent = db.query(Agent).filter(Agent.phone_number == url_encoded_phone).first()
                
                # If still not found, try decoding the phone number (in case DB has encoded version)
                if not agent and phone_number.startswith('%'):
                    decoded_phone = unquote(phone_number)
                    agent = db.query(Agent).filter(Agent.phone_number == decoded_phone).first()
            
            # If no agent found by phone number (or no phone number provided), get the first available agent
            if not agent:
                agent = db.query(Agent).first()
                if agent:
                    print(f"[ContextManager] Using first available agent (namespace: {agent.namespace})")
                    logger.info(f"[ContextManager] Using first available agent (namespace: {agent.namespace})")
            
            if not agent:
                print(f"[ContextManager] No agent found in database")
                logger.error("[ContextManager] No agent found in database")
                return None

            # Fetch tools for agent using tool_ids array from agent
            tool_ids = agent.tool_ids if agent.tool_ids else []
            tools = db.query(Tool).filter(Tool.tool_id.in_(tool_ids)).all() if tool_ids else []

            # Create context
            # Use agent's phone_number if available, otherwise use provided or default
            context_phone = agent.phone_number or phone_number or "default"
            
            context = ConversationContext(
                phone_number=context_phone,
                agent=agent,
                tools=tools,
                websocket=websocket,
            )

            # Store in cache using the context key
            self._contexts[context_key] = context

            return context

        finally:
            db.close()

    def get_context(self, phone_number: str) -> Optional[ConversationContext]:
        """Get context for phone number if it exists."""
        return self._contexts.get(phone_number)

    def remove_context(self, phone_number: str):
        """Remove context for phone number."""
        if phone_number in self._contexts:
            del self._contexts[phone_number]

    def cleanup_expired_contexts(self):
        """Remove expired contexts."""
        expired_phones = []
        for phone_number, context in self._contexts.items():
            if context.is_expired():
                expired_phones.append(phone_number)

        for phone_number in expired_phones:
            self.remove_context(phone_number)

    async def start_cleanup_task(self):
        """Start background task to clean up expired contexts."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background loop to clean up expired contexts."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self.cleanup_expired_contexts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup loop: {e}")

    def close_websocket(self, phone_number: str):
        """Close websocket and mark context as inactive."""
        if phone_number in self._contexts:
            context = self._contexts[phone_number]
            context.websocket = None
            context.last_activity = datetime.now()


# Global instance
context_manager = ContextManager()

