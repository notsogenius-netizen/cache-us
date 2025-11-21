"""
Pydantic schemas for API request/response validation.
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================
# Tool Schemas
# ============================================

class ToolBase(BaseModel):
    """Base tool schema."""
    tool_name: str = Field(..., max_length=100, description="Unique tool name")
    description: Optional[str] = Field(None, description="Tool description")
    parameters: Optional[dict] = Field(None, description="Tool parameter schema (JSON)")


class ToolCreate(ToolBase):
    """Schema for creating a tool."""
    pass


class ToolUpdate(BaseModel):
    """Schema for updating a tool."""
    tool_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolResponse(ToolBase):
    """Schema for tool response."""
    tool_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================
# Agent Schemas
# ============================================

class AgentBase(BaseModel):
    """Base agent schema."""
    namespace: str = Field(..., max_length=255, description="Unique namespace for agent")
    prompt: str = Field(..., description="Base prompt with deterministic tool invocation rules")
    phone_number: Optional[str] = Field(None, max_length=20, description="Twilio phone number")
    webhook_url: Optional[str] = Field(None, max_length=500, description="Webhook URL for callbacks")


class AgentCreate(AgentBase):
    """Schema for creating an agent."""
    tool_ids: Optional[List[UUID]] = Field(default_factory=list, description="List of tool IDs to associate")


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""
    namespace: Optional[str] = Field(None, max_length=255)
    prompt: Optional[str] = None
    phone_number: Optional[str] = Field(None, max_length=20)
    webhook_url: Optional[str] = Field(None, max_length=500)
    tool_ids: Optional[List[UUID]] = None


class AgentResponse(AgentBase):
    """Schema for agent response with tools."""
    ag_id: UUID
    tools: List[ToolResponse] = Field(default_factory=list, description="List of associated tools")
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================
# Agent-Tool Relationship Schemas
# ============================================

class AgentToolResponse(BaseModel):
    """Schema for agent-tool relationship response."""
    agent_id: UUID
    tool_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

