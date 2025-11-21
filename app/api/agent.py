"""
Agent API endpoints.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.schemas import AgentResponse
from app.services.agent_service import (
    process_csv_and_create_embeddings,
    save_faiss_index,
    create_agent_with_validation,
)

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/", response_model=AgentResponse, status_code=201)
async def create_agent(
    prompt: str = Form(..., description="Base prompt for the agent"),
    csv_file: UploadFile = File(..., description="CSV file containing knowledge base data"),
    tool_ids: Optional[str] = Form(None, description="Comma-separated list of tool UUIDs"),
    db: Session = Depends(get_db),
):
    """
    Create a new agent with knowledge base from CSV file.
    
    This endpoint:
    1. Processes the CSV file and generates embeddings
    2. Saves embeddings to FAISS index
    3. Creates agent record in database
    4. Associates tools with the agent
    
    Args:
        prompt: Base prompt for the agent
        csv_file: CSV file containing data to embed
        tool_ids: Optional comma-separated list of tool UUIDs
        db: Database session
        
    Returns:
        Created agent with associated tools
    """
    # Validate CSV file
    if not csv_file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV file"
        )
    
    # Generate namespace from CSV filename (remove .csv extension)
    namespace = csv_file.filename.rsplit('.csv', 1)[0]
    if not namespace:
        # Fallback: generate namespace if filename is invalid
        from uuid import uuid4
        namespace = f"{settings.default_agent_namespace_prefix}{uuid4().hex[:8]}"
    
    # Parse tool IDs if provided
    parsed_tool_ids: Optional[List[UUID]] = None
    if tool_ids:
        try:
            parsed_tool_ids = [UUID(tid.strip()) for tid in tool_ids.split(',') if tid.strip()]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tool ID format: {str(e)}"
            )
    
    # Process CSV and create embeddings
    try:
        text_chunks, embeddings = process_csv_and_create_embeddings(csv_file)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing CSV file: {str(e)}"
        )
    
    # Save FAISS index
    try:
        save_faiss_index(namespace, embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving FAISS index: {str(e)}"
        )
    
    # Create agent in database
    try:
        agent = create_agent_with_validation(
            db=db,
            prompt=prompt,
            namespace=namespace,
            tool_ids=parsed_tool_ids
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating agent: {str(e)}"
        )
    
    # Refresh to load data
    db.refresh(agent)
    
    # Return agent response (tool_ids are now stored directly in the agent)
    return agent

