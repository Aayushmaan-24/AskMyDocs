"""

ingestion.py - PDF loading + chunking strategies
Strategies: fixed-size, recursive, sentence-aware

"""

import os
import json
import hashlib
from pathlib import Path
from typing import Literal
from pypdf import PdfReader
from rich.console import Console
from rich.progress import track

console = Console()

ChunkStrategy = Literal["fixed", "recursive", "sentence"]

# ── 1. PDF → raw text ──────────────────────────────────────────────

def load_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page, preserving page metadata"""
    reader = PdfReader(pdf_path)
    pages = []
    for i , page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({
                "page" : i+1,
                "text" : text,
                "source" : Path(pdf_path).name,
            })
            
    return pages


# ── 2. Chunking strategies ─────────────────────────────────────────

def chunk_fixed(pages: list[dict], size: int = 512, overlap: int = 64) -> list[dict]:
    """Fixed size character chunking with overlap"""
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text" : chunk_text,
                    "source" : page['source'],
                    "page" : page['page'],
                    "strategy" : "fixed",
                    "chunk_id" : _make_id(chunk_text),
                })
            start += size - overlap
    return chunks


