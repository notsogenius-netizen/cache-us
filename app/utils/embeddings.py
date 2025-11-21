"""
Embedding utility for generating embeddings from text data.
"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingModel:
    """Singleton class to manage the embedding model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
        return cls._instance
    
    def get_model(self) -> SentenceTransformer:
        """Get or load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(settings.embedding_model_name)
        return self._model


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        numpy array of shape (n_texts, embedding_dimension) containing embeddings
    """
    model = EmbeddingModel().get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text string to embed
        
    Returns:
        numpy array of shape (embedding_dimension,) containing the embedding
    """
    model = EmbeddingModel().get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

