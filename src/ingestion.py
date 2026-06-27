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

# ── 3. Deduplication ───────────────────────────────────────────────

def deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove exact-duplicate chunks by content hash."""
    seen = set()
    unique = []
    for chunk in chunks:
        if chunk['chunk_id'] not in seen:
            seen.add(chunk['chunk_id'])
            unique.append(chunk)
    return unique

# ── 4. Save / Load ─────────────────────────────────────────────────

def save_chunks(chunks: list[dict], path: str='data/chunks/chunks.json'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dumps(chunks, f, indent = 2)
    console.print(f"[green]✓ Saved {len(chunks)} chunks → {path}[/green]")
    
def load_chunks(path: str = "data/chunks/chunks.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)
    
# ── 5. Main: run all three strategies + compare ────────────────────

def ingest_pdfs(pdf_dir: str = "data/pdfs", strategy : ChunkStrategy = "recursive") -> list[dict]:
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDFs found in {pdf_dir}[/red]")
        return []
    
    all_pages = []
    for pdf in track(pdf_files, description="Loading PDFs..."):
        pages = load_pdf(str(pdf))
        all_pages.extend(pages)
        console.print(f"  [cyan]{pdf.name}[/cyan] → {len(pages)} pages")
        
    console.print(f"\n[bold]Total pages loaded:[/bold] {len(all_pages)}")
    
    chunkers = {
        "fixed" : lambda p : chunk_fixed(p),
        "recursive" : lambda p : chunk_recursive(p),
        "sentence" : lambda p: chunk_sentence(p),
    }
    
    # comparisons across all three
    console.print("\n[bold yellow]── Chunking Strategy Comparison ──[/bold yellow]")
    for name, fn in chunkers.items():
        result = deduplicate(fn(all_pages))
        avg_len = sum(len(c['text']) for c in result) / len(result) if result else 0
        console.print(f"  {name:12s} → {len(result):5d} chunks  |  avg {avg_len:6.0f} chars")
        
    # use chosen strategy for final working
    chosen_fn = chunkers[strategy]
    chunks = deduplicate(chosen_fn(all_pages))
    console.print(f"\n[green]Using strategy:[/green] [bold]{strategy}[/bold] → {len(chunks)} chunks")
    
    save_chunks(chunks)
    return chunks
