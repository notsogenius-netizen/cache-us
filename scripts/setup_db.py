#!/usr/bin/env python3
"""
Simple script to set up database tables.
Run this once to create all tables in PostgreSQL.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.init_db import init_db

if __name__ == "__main__":
    init_db()

