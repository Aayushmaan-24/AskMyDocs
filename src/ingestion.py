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
import re

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


def chunk_recursive(pages: list[dict], target_size: int = 512, overlap: int = 64) -> list[dict]:
    """Recursive splitting: paragraph → sentence → word fallback."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    
    def split_text(text: str, sep_idx: int = 0) -> list[str]:
        if len(text) <= target_size or sep_idx >= len(separators):
            return [text]
        sep = separators[sep_idx]
        parts = text.split(sep)
        result = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= target_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > target_size:
                    result.extend(split_text(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result
    
    for page in pages:
        parts = split_text(page["text"])
        for part in parts:
            part = part.strip()
            if len(part) > 50:
                chunks.append({
                    "text": part,
                    "source": page["source"],
                    "page":page["page"],
                    "strategy": "recursive",
                    "chunk_id": _make_id(part),
                })
    return chunks

def chunk_sentence(pages: list[dict], sentences_per_chunk: int = 5, overlap: int = 1) -> list[dict]:
    """Sentence-aware chunking — keeps semantic units intact."""
    chunks = []
    sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    
    for page in pages:
        sentences = sentence_splitter.split(page['text'])
        sentences = [s.strip() for s in sentences if len(s.strip())>20]
        i = 0
        while i < len(sentences):
            window = sentences[i: i+sentences_per_chunk]
            chunk_text = " ".join(window).strip()
            if chunk_text:
                chunks.append({
                    "text" : chunk_text,
                    "source" : page['source'],
                    "page" : page['page'],
                    "strategy" : "sentence",
                    "chunk_id" : _make_id(chunk_text),
                })
            i += sentences_per_chunk - overlap
    return chunks

