"""
retrieval_service.py
--------------------
Orchestrates the full document ingestion pipeline AND the query retrieval step.

Ingestion pipeline:
  PDF file → extract text → split into chunks → embed chunks → store in FAISS

Retrieval pipeline:
  User query → embed query → search FAISS → return top-K relevant chunks
"""

import os
from typing import List, Tuple

from utils.pdf_loader import PDFLoader
from utils.text_chunker import TextChunker
from services.embedding_service import EmbeddingService
from database.vector_store import VectorStore


class RetrievalService:
    """End-to-end document ingestion and semantic retrieval."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, top_k: int = 4):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Shared services
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Track which document is currently loaded
        self.current_doc_name: str = None

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def ingest_pdf(self, file_path: str) -> dict:
        """
        Full pipeline: load PDF → chunk → embed → index.

        Args:
            file_path: Absolute path to the uploaded PDF.

        Returns:
            Summary dict with chunk count, page count, etc.
        """
        print(f"[RetrievalService] Ingesting: {file_path}")

        # Step 1: Extract text
        loader = PDFLoader(file_path)
        text = loader.extract_text()
        metadata = loader.get_metadata()

        # Step 2: Split into chunks
        chunk_dicts = self.chunker.split_with_metadata(
            text, source=os.path.basename(file_path)
        )
        chunk_texts = [c["text"] for c in chunk_dicts]

        # Step 3: Embed all chunks
        embeddings = self.embedder.embed_chunks(chunk_texts)

        # Step 4: Store in FAISS (clear any previous document first)
        self.vector_store.clear()
        self.vector_store.build_index(embeddings, chunk_dicts)

        self.current_doc_name = metadata["file"]

        return {
            "file": metadata["file"],
            "pages": metadata["pages"],
            "chunks": len(chunk_dicts),
            "embedding_dim": self.embedder.embedding_dim,
            "status": "indexed",
        }

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> List[Tuple[dict, float]]:
        """
        Embed a user query and return the top-K most relevant chunks.

        Args:
            query: The user's natural language question.

        Returns:
            List of (chunk_dict, similarity_score) tuples.
        """
        if not self.vector_store.is_ready:
            raise RuntimeError("No document has been indexed yet. Please upload a PDF first.")

        query_vector = self.embedder.embed_text(query)
        results = self.vector_store.search(query_vector, top_k=self.top_k)
        return results

    def get_context_string(self, query: str) -> Tuple[str, List[dict]]:
        """
        Convenience method: retrieve chunks and format them as a single
        context string ready to be injected into the LLM prompt.

        Returns:
            (context_string, list_of_source_chunks)
        """
        results = self.retrieve(query)
        context_parts = []
        sources = []

        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(f"[Excerpt {i} | Score: {score}]\n{chunk['text']}")
            sources.append({"chunk_id": chunk["chunk_id"], "score": score, "source": chunk["source"]})

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready
