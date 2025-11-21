"""
FAISS engine for loading and caching FAISS indices by namespace.
"""
import os
import faiss
from typing import Optional, Dict
from pathlib import Path

from app.core.config import settings


class FAISSEngine:
    """Engine for managing FAISS indices by namespace."""

    def __init__(self):
        self._indices_cache: Dict[str, faiss.Index] = {}
        self.index_dir = Path(settings.faiss_index_dir)
        self.embedding_dim = settings.embedding_dimension
        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def get_index_path(self, namespace: str) -> Path:
        """
        Get the path to FAISS index for a namespace.

        Args:
            namespace: Agent namespace

        Returns:
            Path to FAISS index file
        """
        namespace_dir = self.index_dir / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)
        return namespace_dir / "index.faiss"

    def load_index(self, namespace: str) -> Optional[faiss.Index]:
        """
        Load FAISS index for a namespace (cached if already loaded).

        Args:
            namespace: Agent namespace

        Returns:
            FAISS index object or None if not found
        """
        # Return cached index if available
        if namespace in self._indices_cache:
            return self._indices_cache[namespace]

        # Load index from disk
        index_path = self.get_index_path(namespace)
        
        if not index_path.exists():
            # Index doesn't exist, return None
            return None

        try:
            # Load FAISS index
            index = faiss.read_index(str(index_path))
            # Cache it
            self._indices_cache[namespace] = index
            return index
        except Exception as e:
            # Log error if needed
            return None

    def get_or_load_index(self, namespace: str) -> Optional[faiss.Index]:
        """
        Get cached index or load it if not cached.

        Args:
            namespace: Agent namespace

        Returns:
            FAISS index object or None if not found
        """
        return self.load_index(namespace)

    def remove_from_cache(self, namespace: str):
        """
        Remove index from cache (but keep on disk).

        Args:
            namespace: Agent namespace
        """
        if namespace in self._indices_cache:
            del self._indices_cache[namespace]

    def clear_cache(self):
        """Clear all cached indices from memory."""
        self._indices_cache.clear()


# Global instance
faiss_engine = FAISSEngine()

