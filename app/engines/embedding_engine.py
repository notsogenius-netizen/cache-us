"""
Embedding engine for text embeddings using sentence-transformers.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingEngine:
    """Engine for generating text embeddings."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._model = SentenceTransformer(settings.embedding_model_name)

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            numpy array of embedding (384-dim)
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.reshape(1, -1)  # Shape: (1, 384)

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings (n, 384)
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings


# Global instance
embedding_engine = EmbeddingEngine()

