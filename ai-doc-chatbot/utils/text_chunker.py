"""
text_chunker.py
---------------
Splits a large text into smaller overlapping chunks.

Why chunking?
  - LLMs have token limits, so we can't send 100 pages at once.
  - Smaller chunks make similarity search more precise.
  - Overlap ensures context isn't lost at chunk boundaries.

Example:
  Text: "The cat sat on the mat. The mat was red."
  Chunk size 5 words, overlap 2:
    Chunk 1: "The cat sat on the"
    Chunk 2: "on the mat. The mat"
    Chunk 3: "The mat was red."
"""

from typing import List


class TextChunker:
    """Splits text into overlapping chunks by word count."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size:    Number of words per chunk.
            chunk_overlap: Number of words shared between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """
        Split text into overlapping word-based chunks.
        Returns a list of chunk strings.
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text.")

        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap  # how far we advance each time

        for start in range(0, len(words), step):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            # Stop if we've passed the end
            if end >= len(words):
                break

        return chunks

    def split_with_metadata(self, text: str, source: str = "") -> List[dict]:
        """
        Same as split() but each chunk is returned as a dict with metadata.
        Useful for storing alongside embeddings.
        """
        chunks = self.split(text)
        return [
            {
                "chunk_id": i,
                "text": chunk,
                "source": source,
                "word_count": len(chunk.split()),
            }
            for i, chunk in enumerate(chunks)
        ]
