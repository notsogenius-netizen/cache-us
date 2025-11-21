"""
Join table for agent-tool relationships (many-to-many).
"""
from sqlalchemy import Column, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class AgentTool(Base):
    """Join table linking agents to their tools."""

    __tablename__ = "agent_tools"

    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.ag_id", ondelete="CASCADE"),
        primary_key=True,
    )
    tool_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tools.tool_id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    agent = relationship("Agent", back_populates="tools")
    tool = relationship("Tool")

    def __repr__(self):
        return f"<AgentTool(agent_id={self.agent_id}, tool_id={self.tool_id})>"

