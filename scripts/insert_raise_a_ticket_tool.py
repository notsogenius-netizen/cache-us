"""
Script to insert the raise-a-ticket tool into the database.
"""
import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from app.core.database import engine
from app.models.tool import Tool
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
import uuid

def insert_raise_a_ticket_tool():
    """Insert the raise-a-ticket tool into the database."""
    db = SessionLocal()
    try:
        # Check if tool already exists
        existing_tool = db.query(Tool).filter(Tool.tool_name == 'raise-a-ticket').first()
        
        if existing_tool:
            print(f"Tool 'raise-a-ticket' already exists with ID: {existing_tool.tool_id}")
            # Update it anyway
            existing_tool.description = 'Creates a support ticket by posting to http://localhost:3000/raise-a-ticket. Use this whenever the user reports any issue. The reason should be a clear and descriptive summary of the problem the user is facing.'
            existing_tool.parameters = {
                "curl": 'curl -X POST "http://localhost:3000/raise-a-ticket" -H "Content-Type: application/json" -d \'{}\''
            }
            db.commit()
            print("✓ Updated existing 'raise-a-ticket' tool")
        else:
            # Create new tool
            new_tool = Tool(
                tool_id=uuid.uuid4(),
                tool_name='raise-a-ticket',
                description='Creates a support ticket by posting to http://localhost:3000/raise-a-ticket. Use this whenever the user reports any issue. The reason should be a clear and descriptive summary of the problem the user is facing.',
                parameters={
                    "curl": 'curl -X POST "http://localhost:3000/raise-a-ticket" -H "Content-Type: application/json" -d \'{}\''
                }
            )
            db.add(new_tool)
            db.commit()
            print(f"✓ Created 'raise-a-ticket' tool with ID: {new_tool.tool_id}")
        
        # Query and display the tool
        tool = db.query(Tool).filter(Tool.tool_name == 'raise-a-ticket').first()
        print(f"\nTool Details:")
        print(f"  ID: {tool.tool_id}")
        print(f"  Name: {tool.tool_name}")
        print(f"  Description: {tool.description[:80]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Inserting 'raise-a-ticket' tool into database...")
    insert_raise_a_ticket_tool()
    print("\n✅ Done!")

