"""
RAG service for querying FAISS and retrieving relevant context chunks.
"""
from typing import List, Optional
import numpy as np

from app.engines.faiss_engine import faiss_engine
from app.engines.embedding_engine import embedding_engine


class RAGService:
    """Service for RAG (Retrieval Augmented Generation) queries."""

    async def get_rag_context(
        self, transcribed_text: str, namespace: str, top_k: int = 3
    ) -> List[str]:
        """
        Query FAISS for relevant context chunks (fresh query, no caching of results).

        Args:
            transcribed_text: User's transcribed text
            namespace: Agent namespace to identify which FAISS index to use
            top_k: Number of top chunks to retrieve (default: 3)

        Returns:
            List of relevant text chunks (strings)
        """
        # Get or load FAISS index for namespace (index is cached, not query results)
        index = faiss_engine.get_or_load_index(namespace)
        
        if index is None:
            # No index found for this namespace, return empty list
            return []

        try:
            # Generate embedding for transcribed text
            query_embedding = await embedding_engine.embed_text(transcribed_text)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            # Query FAISS index (fresh query - no result caching)
            k = min(top_k, index.ntotal)  # Ensure k doesn't exceed total vectors
            if k == 0:
                return []

            distances, indices = index.search(query_embedding, k)

            # Retrieve text chunks from metadata storage
            # TODO: Implement actual chunk loading based on your storage mechanism
            # This depends on how you store text chunks alongside FAISS indices.
            # Common approaches:
            # 1. Separate metadata file (JSON/Pickle) mapping indices to chunks
            # 2. Database table with chunk text
            # 3. Stored alongside FAISS index in same directory
            
            # For now, returning placeholder chunks
            # You'll need to implement: chunks = load_chunks_from_storage(namespace, indices[0])
            chunks = []
            
            # Example implementation structure:
            # metadata_file = faiss_engine.get_index_path(namespace).parent / "metadata.json"
            # if metadata_file.exists():
            #     with open(metadata_file) as f:
            #         metadata = json.load(f)
            #     chunks = [metadata[str(idx)] for idx in indices[0] if idx >= 0 and str(idx) in metadata]
            
            # Placeholder: return indices as strings for now
            for idx in indices[0]:
                if idx >= 0:  # Valid index
                    chunks.append(f"Context chunk {idx}")  # Replace with actual chunk loading

            return chunks

        except Exception as e:
            # Log error and return empty list
            print(f"Error querying FAISS: {e}")
            return []


# Global instance
rag_service = RAGService()

