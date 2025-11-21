"""
Agent service for managing agent creation and operations.
"""
import csv
import io
import numpy as np
from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException

from app.core.config import settings
from app.core.database import get_db
from app.models.agent import Agent
from app.models.tool import Tool
from app.utils.embeddings import generate_embeddings
from app.adapters.faiss_adapter import FAISSAdapter


def process_csv_and_create_embeddings(csv_file: UploadFile) -> Tuple[List[str], np.ndarray]:
    """
    Read CSV file and generate embeddings for the data.
    
    Args:
        csv_file: Uploaded CSV file
        
    Returns:
        Tuple of (text_chunks, embeddings) where:
        - text_chunks: List of text strings extracted from CSV
        - embeddings: numpy array of embeddings
    """
    # Read CSV content
    content = csv_file.file.read()
    csv_file.file.seek(0)  # Reset file pointer
    
    # Decode content
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    
    # Parse CSV
    csv_reader = csv.DictReader(io.StringIO(content))
    
    # Extract text from all columns and combine into chunks
    text_chunks = []
    for row in csv_reader:
        # Combine all column values into a single text string
        row_text = " ".join(str(value) for value in row.values() if value)
        if row_text.strip():  # Only add non-empty rows
            text_chunks.append(row_text.strip())
    
    if not text_chunks:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or contains no valid data"
        )
    
    # Generate embeddings
    embeddings = generate_embeddings(text_chunks)
    
    return text_chunks, embeddings


def save_faiss_index(namespace: str, embeddings: np.ndarray):
    """
    Save embeddings to FAISS index.
    
    Args:
        namespace: Unique namespace for the index
        embeddings: numpy array of embeddings to save
    """
    faiss_adapter = FAISSAdapter()
    
    # Create index from embeddings
    index = faiss_adapter.create_index(embeddings)
    
    # Save index
    faiss_adapter.save_index(namespace, index)


def create_agent_with_validation(
    db: Session,
    prompt: str,
    namespace: str,
    tool_ids: Optional[List[UUID]] = None
) -> Agent:
    """
    Create an agent with validation checks.
    
    Args:
        db: Database session
        prompt: Base prompt for the agent
        namespace: Unique namespace for the agent
        tool_ids: Optional list of tool IDs to associate
        
    Returns:
        Created Agent object
        
    Raises:
        HTTPException: If validation fails
    """
    # Check if phone_number already exists in DB
    phone_number = settings.twilio_phone_number
    if phone_number:
        existing_agent = db.query(Agent).filter(
            Agent.phone_number == phone_number
        ).first()
        
        if existing_agent:
            raise HTTPException(
                status_code=400,
                detail=f"Agent with phone number {phone_number} already exists. Only one agent can use this phone number."
            )
    
    # Check if namespace already exists
    existing_namespace = db.query(Agent).filter(
        Agent.namespace == namespace
    ).first()
    
    if existing_namespace:
        raise HTTPException(
            status_code=400,
            detail=f"Agent with namespace '{namespace}' already exists"
        )
    
    # Validate tools if provided
    if tool_ids:
        tools = db.query(Tool).filter(Tool.tool_id.in_(tool_ids)).all()
        if len(tools) != len(tool_ids):
            found_ids = {tool.tool_id for tool in tools}
            missing_ids = set(tool_ids) - found_ids
            raise HTTPException(
                status_code=404,
                detail=f"Tools not found: {list(missing_ids)}"
            )
        
        # Check tool limit
        if len(tool_ids) > settings.max_tools_per_agent:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {settings.max_tools_per_agent} tools allowed per agent"
            )
    
    # Get webhook URL from settings
    webhook_url = settings.twilio_webhook_base_url
    
    # Create agent with tool_ids stored directly
    agent = Agent(
        namespace=namespace,
        prompt=prompt,
        phone_number=phone_number,
        webhook_url=webhook_url,
        tool_ids=tool_ids if tool_ids else []
    )
    
    db.add(agent)
    db.commit()
    db.refresh(agent)
    
    return agent

