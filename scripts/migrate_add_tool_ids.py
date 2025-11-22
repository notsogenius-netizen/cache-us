#!/usr/bin/env python3
"""
Migration script to add tool_ids column to agents table.
This script adds the missing tool_ids column that was added to the Agent model.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, inspect
from app.core.database import engine
from app.core.config import settings


def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def migrate_add_tool_ids():
    """
    Add tool_ids column to agents table if it doesn't exist.
    """
    print("Checking agents table schema...")
    
    # Check if column already exists
    if check_column_exists('agents', 'tool_ids'):
        print("✓ Column 'tool_ids' already exists in 'agents' table. No migration needed.")
        return
    
    print("Adding 'tool_ids' column to 'agents' table...")
    
    # Add the column using raw SQL
    # tool_ids is an ARRAY of UUIDs, nullable, with default empty array
    with engine.connect() as conn:
        # Start a transaction
        trans = conn.begin()
        try:
            # Add the column
            conn.execute(text("""
                ALTER TABLE agents 
                ADD COLUMN tool_ids UUID[] DEFAULT '{}'::UUID[]
            """))
            trans.commit()
            print("✓ Successfully added 'tool_ids' column to 'agents' table!")
        except Exception as e:
            trans.rollback()
            print(f"✗ Error adding column: {e}")
            raise


if __name__ == "__main__":
    try:
        migrate_add_tool_ids()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

