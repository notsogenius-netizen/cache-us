#!/usr/bin/env python3
"""
Migration script to fix phone numbers in the database.
Converts URL-encoded phone numbers (%2B...) to decoded format (+...)
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from urllib.parse import unquote
from sqlalchemy import text
from app.core.database import engine
from app.core.config import settings


def fix_phone_numbers():
    """
    Fix phone numbers in agents table by decoding URL-encoded values.
    """
    print("Checking and fixing phone numbers in agents table...")
    
    with engine.connect() as conn:
        # Start a transaction
        trans = conn.begin()
        try:
            # Get all agents with phone numbers
            result = conn.execute(text("""
                SELECT ag_id, phone_number 
                FROM agents 
                WHERE phone_number IS NOT NULL
            """))
            
            agents_to_fix = []
            for row in result:
                ag_id, phone_number = row
                # Check if phone number is URL-encoded (starts with %)
                if phone_number and phone_number.startswith('%'):
                    decoded = unquote(phone_number)
                    agents_to_fix.append((ag_id, phone_number, decoded))
                    print(f"  Found URL-encoded phone: {phone_number} -> {decoded}")
            
            if not agents_to_fix:
                print("✓ No URL-encoded phone numbers found. All phone numbers are in correct format.")
                trans.commit()
                return
            
            # Update each agent
            for ag_id, encoded, decoded in agents_to_fix:
                conn.execute(text("""
                    UPDATE agents 
                    SET phone_number = :decoded 
                    WHERE ag_id = :ag_id
                """), {"decoded": decoded, "ag_id": ag_id})
                print(f"  Updated agent {ag_id}: {encoded} -> {decoded}")
            
            trans.commit()
            print(f"✓ Successfully fixed {len(agents_to_fix)} phone number(s)!")
            
        except Exception as e:
            trans.rollback()
            print(f"✗ Error fixing phone numbers: {e}")
            raise


if __name__ == "__main__":
    try:
        fix_phone_numbers()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

