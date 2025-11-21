"""
Tool model for storing tool definitions.
"""
from sqlalchemy import Column, String, Text, JSON, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class Tool(Base):
    """Tool model representing available tools in the system."""

    __tablename__ = "tools"

    tool_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tool_name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True)  # Tool parameter schema
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Tool(tool_id={self.tool_id}, tool_name={self.tool_name})>"

