"""
Agent model for storing agent configurations.
"""
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class Agent(Base):
    """Agent model representing agent configurations."""

    __tablename__ = "agents"

    ag_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace = Column(String(255), unique=True, nullable=False, index=True)
    prompt = Column(Text, nullable=False)  # Base prompt with tool invocation rules
    phone_number = Column(String(20), nullable=True, index=True)
    webhook_url = Column(String(500), nullable=True)
    tool_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True, default=list)  # List of tool UUIDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Agent(ag_id={self.ag_id}, namespace={self.namespace})>"

