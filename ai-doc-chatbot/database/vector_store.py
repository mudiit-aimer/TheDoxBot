"""
vector_store.py
---------------
Stores and searches embeddings using FAISS.

What is FAISS?
  Facebook AI Similarity Search — a library that can find the most similar
  vectors to a query vector extremely fast, even with millions of vectors.

How it works here:
  1. We build an index from all chunk embeddings when a doc is uploaded.
  2. When user asks a question, we embed the question and search the index.
  3. FAISS returns the indices of the top-K most similar chunks.
  4. We use those indices to fetch the actual text chunks.
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np


class VectorStore:
    """FAISS-backed vector store for document chunk embeddings."""

    def __init__(self, persist_dir: str = "vector_store_data"):
        """
        Args:
            persist_dir: Directory where the FAISS index and chunk texts are saved.
        """
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "faiss.index")
        self.chunks_path = os.path.join(persist_dir, "chunks.pkl")

        # In-memory state
        self.index: faiss.Index = None        # The FAISS index
        self.chunks: List[dict] = []          # The raw text chunks + metadata
        self.is_ready: bool = False           # True once a doc has been indexed

        os.makedirs(persist_dir, exist_ok=True)

        # Auto-load if a saved index already exists
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            self._load()

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def build_index(self, embeddings: np.ndarray, chunks: List[dict]):
        """
        Build a FAISS index from embeddings and store corresponding chunk texts.

        Args:
            embeddings: 2D numpy array of shape (num_chunks, embedding_dim).
            chunks:     List of dicts with at least a 'text' key.
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")

        dim = embeddings.shape[1]

        # IndexFlatL2 = exact nearest-neighbour search using L2 (Euclidean) distance.
        # For cosine similarity, we normalise first then use L2 (equivalent).
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.chunks = chunks
        self.is_ready = True

        self._save()
        print(f"[VectorStore] Index built with {self.index.ntotal} vectors (dim={dim}).")

    # ------------------------------------------------------------------ #
    #  Searching                                                           #
    # ------------------------------------------------------------------ #

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[dict, float]]:
        """
        Find the top-K chunks most similar to the query vector.

        Returns:
            List of (chunk_dict, similarity_score) tuples, best match first.
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is empty. Please upload a document first.")

        # Normalise query so L2 distance ≈ cosine similarity
        query = query_vector.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)

        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:           # FAISS returns -1 when there aren't enough vectors
                continue
            score = float(1 - dist / 2)   # Convert L2 distance → similarity (0–1)
            results.append((self.chunks[idx], round(score, 4)))

        return results

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _save(self):
        """Persist the FAISS index and chunk texts to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorStore] Saved to '{self.persist_dir}'.")

    def _load(self):
        """Load a previously saved index from disk."""
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.is_ready = True
        print(f"[VectorStore] Loaded index with {self.index.ntotal} vectors.")

    def clear(self):
        """Wipe the index (call before indexing a new document)."""
        self.index = None
        self.chunks = []
        self.is_ready = False
        for path in [self.index_path, self.chunks_path]:
            if os.path.exists(path):
                os.remove(path)
        print("[VectorStore] Cleared.")

    @property
    def doc_count(self) -> int:
        return self.index.ntotal if self.is_ready else 0
