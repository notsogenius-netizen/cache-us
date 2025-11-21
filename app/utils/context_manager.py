"""
Context manager for caching conversation context per phone number.
"""
from typing import Dict, Optional, List, Any
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.agent import Agent
from app.models.tool import Tool
from app.models.agent_tool import AgentTool
from app.core.database import SessionLocal
from app.core.config import settings


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
        self, phone_number: str, websocket: Any
    ) -> Optional[ConversationContext]:
        """
        Get existing context or create new one for phone number.

        Args:
            phone_number: Phone number
            websocket: WebSocket connection

        Returns:
            ConversationContext or None if agent not found
        """
        # Check if context exists
        if phone_number in self._contexts:
            context = self._contexts[phone_number]
            context.websocket = websocket  # Update websocket
            context.last_activity = datetime.now()
            return context

        # Create new context - fetch agent from database
        db = SessionLocal()
        try:
            # Fetch agent by phone number
            agent = db.query(Agent).filter(Agent.phone_number == phone_number).first()
            
            if not agent:
                return None

            # Fetch tools for agent
            agent_tools = (
                db.query(AgentTool)
                .filter(AgentTool.agent_id == agent.ag_id)
                .all()
            )
            
            tool_ids = [at.tool_id for at in agent_tools]
            tools = db.query(Tool).filter(Tool.tool_id.in_(tool_ids)).all() if tool_ids else []

            # Create context
            context = ConversationContext(
                phone_number=phone_number,
                agent=agent,
                tools=tools,
                websocket=websocket,
            )

            # Store in cache
            self._contexts[phone_number] = context

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

