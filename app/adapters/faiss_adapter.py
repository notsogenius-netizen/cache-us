"""
FAISS adapter for managing vector indices.
"""
import os
import faiss
import numpy as np
from pathlib import Path
from typing import Optional

from app.core.config import settings


class FAISSAdapter:
    """Adapter for managing FAISS vector indices."""
    
    def __init__(self):
        """Initialize FAISS adapter with configuration."""
        self.index_dir = Path(settings.faiss_index_dir)
        self.embedding_dim = settings.embedding_dimension
        self._ensure_index_dir()
    
    def _ensure_index_dir(self):
        """Ensure the index directory exists."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create a new FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (n_vectors, embedding_dim)
            
        Returns:
            FAISS index object
        """
        # Create a flat L2 index (can be upgraded to IVFFlat or other types later)
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity (optional, but common practice)
        # For now, we'll use L2 distance without normalization
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        return index
    
    def save_index(self, namespace: str, index: faiss.Index):
        """
        Save a FAISS index to disk.
        
        Args:
            namespace: Unique namespace/identifier for the index
            index: FAISS index object to save
        """
        index_path = self.index_dir / f"{namespace}.index"
        faiss.write_index(index, str(index_path))
    
    def load_index(self, namespace: str) -> Optional[faiss.Index]:
        """
        Load a FAISS index from disk.
        
        Args:
            namespace: Unique namespace/identifier for the index
            
        Returns:
            FAISS index object or None if not found
        """
        index_path = self.index_dir / f"{namespace}.index"
        
        if not index_path.exists():
            return None
        
        return faiss.read_index(str(index_path))
    
    def index_exists(self, namespace: str) -> bool:
        """
        Check if an index exists for a given namespace.
        
        Args:
            namespace: Unique namespace/identifier for the index
            
        Returns:
            True if index exists, False otherwise
        """
        index_path = self.index_dir / f"{namespace}.index"
        return index_path.exists()
    
    def delete_index(self, namespace: str) -> bool:
        """
        Delete a FAISS index from disk.
        
        Args:
            namespace: Unique namespace/identifier for the index
            
        Returns:
            True if deleted, False if not found
        """
        index_path = self.index_dir / f"{namespace}.index"
        
        if index_path.exists():
            index_path.unlink()
            return True
        return False

