"""
embedding_service.py
--------------------
Converts text into numerical vectors (embeddings).

What is an embedding?
  A list of ~384 numbers that represents the MEANING of a sentence.
  Similar sentences have vectors that are "close" to each other in space.
  This is how we find relevant chunks — by finding vectors close to the query vector.

We use sentence-transformers (runs locally, no API key needed).
Model: all-MiniLM-L6-v2  — fast, small, good quality.
"""

from typing import List
import numpy as np


class EmbeddingService:
    """Generates sentence embeddings using a local transformer model."""

    # Class-level cache so model loads only once across requests
    _model = None
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        """Load the embedding model (cached after first load)."""
        if EmbeddingService._model is None:
            print(f"[EmbeddingService] Loading model '{self.MODEL_NAME}'... (first time only)")
            from sentence_transformers import SentenceTransformer
            EmbeddingService._model = SentenceTransformer(self.MODEL_NAME)
            print("[EmbeddingService] Model loaded.")
        self.model = EmbeddingService._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single string into a 1D numpy embedding vector.
        Used for embedding the user's query at search time.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.astype("float32")

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Convert a list of text chunks into a 2D numpy array of embeddings.
        Shape: (num_chunks, embedding_dim) e.g. (42, 384)
        Used when ingesting a new document.
        """
        if not chunks:
            raise ValueError("No chunks provided to embed.")
        print(f"[EmbeddingService] Embedding {len(chunks)} chunks...")
        vectors = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype("float32")

    @property
    def embedding_dim(self) -> int:
        """Return the size of each embedding vector (e.g. 384)."""
        return self.model.get_sentence_embedding_dimension()
