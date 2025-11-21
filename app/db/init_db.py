"""
Database initialization script.
Automatically creates all tables in PostgreSQL.
"""
from app.core.database import Base, engine
from app.models import Agent, Tool  # Import all models


def init_db():
    """
    Create all database tables automatically.
    This will create tables for all models defined in app.models.
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully!")


if __name__ == "__main__":
    init_db()

