"""
pdf_loader.py
-------------
Responsible for loading and extracting raw text from PDF files.
Uses PyPDF to read each page and concatenate the text.
"""

import os
from pypdf import PdfReader


class PDFLoader:
    """Handles PDF file reading and text extraction."""

    def __init__(self, file_path: str):
        """
        Initialize with the path to the PDF file.
        Raises FileNotFoundError if the file doesn't exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")
        self.file_path = file_path

    def extract_text(self) -> str:
        """
        Read every page of the PDF and return all text as one string.
        Pages are separated by newlines for clean chunking later.
        """
        reader = PdfReader(self.file_path)

        if len(reader.pages) == 0:
            raise ValueError("PDF has no pages.")

        all_text = []
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Tag each page so we can reference it later if needed
                all_text.append(f"[Page {page_number + 1}]\n{text.strip()}")

        if not all_text:
            raise ValueError("Could not extract any text from PDF. It may be scanned/image-based.")

        return "\n\n".join(all_text)

    def get_metadata(self) -> dict:
        """Return basic PDF metadata (title, author, pages)."""
        reader = PdfReader(self.file_path)
        meta = reader.metadata or {}
        return {
            "title": meta.get("/Title", "Unknown"),
            "author": meta.get("/Author", "Unknown"),
            "pages": len(reader.pages),
            "file": os.path.basename(self.file_path),
        }
