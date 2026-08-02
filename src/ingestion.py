"""

ingestion.py - PDF, DOCX, TXT, MD loading + chunking strategies
Strategies: fixed-size, recursive, sentence-aware

"""

import os
import json
import hashlib
from pathlib import Path
from typing import Literal
from pypdf import PdfReader
import docx
from rich.console import Console
from rich.progress import track
import re

console = Console()

ChunkStrategy = Literal["fixed", "recursive", "sentence"]

# ── 1. Document Loading Helpers ─────────────────────────────────────

def load_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page, preserving page metadata"""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({
                "page": i + 1,
                "text": text,
                "source": Path(pdf_path).name,
            })
    return pages


def load_docx(docx_path: str) -> list[dict]:
    """Extract text from docx file, treating paragraphs as content sections or page slices."""
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    # We can group paragraphs into blocks of roughly 1500 chars to simulate "pages"
    # or just treat the whole document as page 1. Grouping into blocks makes chunking more logical.
    pages = []
    current_block = []
    current_len = 0
    page_num = 1

    for para in full_text:
        current_block.append(para)
        current_len += len(para)
        if current_len >= 1500:
            pages.append({
                "page": page_num,
                "text": "\n\n".join(current_block),
                "source": Path(docx_path).name,
            })
            page_num += 1
            current_block = []
            current_len = 0

    if current_block:
        pages.append({
            "page": page_num,
            "text": "\n\n".join(current_block),
            "source": Path(docx_path).name,
        })

    return pages


def load_txt(txt_path: str) -> list[dict]:
    """Extract text from plain text file, splitting into virtual pages of ~1500 characters."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Clean whitespace
    text = re.sub(r'\r\n', '\n', text)

    # Split by paragraphs or virtual pages
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pages = []
    current_block = []
    current_len = 0
    page_num = 1

    for para in paragraphs:
        current_block.append(para)
        current_len += len(para)
        if current_len >= 1500:
            pages.append({
                "page": page_num,
                "text": "\n\n".join(current_block),
                "source": Path(txt_path).name,
            })
            page_num += 1
            current_block = []
            current_len = 0
            
    if current_block:
        pages.append({
            "page": page_num,
            "text": "\n\n".join(current_block),
            "source": Path(txt_path).name,
        })

    return pages


def load_md(md_path: str) -> list[dict]:
    """Extract text from markdown files, splitting into virtual pages."""
    return load_txt(md_path)


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
                    "text": chunk_text,
                    "source": page['source'],
                    "page": page['page'],
                    "strategy": "fixed",
                    "chunk_id": _make_id(chunk_text),
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
                    "page": page["page"],
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
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        i = 0
        while i < len(sentences):
            window = sentences[i: i + sentences_per_chunk]
            chunk_text = " ".join(window).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": page['source'],
                    "page": page['page'],
                    "strategy": "sentence",
                    "chunk_id": _make_id(chunk_text),
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

def save_chunks(chunks: list[dict], path: str = 'data/chunks/chunks.json'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(chunks, f, indent=2)
    console.print(f"[green]✓ Saved {len(chunks)} chunks → {path}[/green]")
    
def load_chunks(path: str = "data/chunks/chunks.json") -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ── 5. Main: run all strategies + compare ──────────────────────────

def ingest_documents(pdf_dir: str = "data/pdfs", strategy: ChunkStrategy = "recursive", chunk_size: int = 512, chunk_overlap: int = 64) -> list[dict]:
    """Ingests all supported documents (PDF, DOCX, TXT, MD) from target directory."""
    pdf_path_obj = Path(pdf_dir)
    pdf_path_obj.mkdir(parents=True, exist_ok=True)

    # List all files matching our patterns
    patterns = ["*.pdf", "*.docx", "*.txt", "*.md"]
    doc_files = []
    for pat in patterns:
        doc_files.extend(list(pdf_path_obj.glob(pat)))

    if not doc_files:
        console.print(f"[red]No supported files found in {pdf_dir}[/red]")
        save_chunks([])
        return []
    
    all_pages = []
    for doc_file in track(doc_files, description="Loading Documents..."):
        ext = doc_file.suffix.lower()
        if ext == ".pdf":
            pages = load_pdf(str(doc_file))
        elif ext == ".docx":
            pages = load_docx(str(doc_file))
        elif ext in [".txt", ".text"]:
            pages = load_txt(str(doc_file))
        elif ext == ".md":
            pages = load_md(str(doc_file))
        else:
            console.print(f"[yellow]Skipping unsupported file extension: {doc_file.name}[/yellow]")
            continue

        all_pages.extend(pages)
        console.print(f"  [cyan]{doc_file.name}[/cyan] → {len(pages)} sections/pages")
        
    console.print(f"\n[bold]Total pages/sections loaded:[/bold] {len(all_pages)}")
    
    # Custom closure chunkers that can respect chunk_size/chunk_overlap
    chunkers = {
        "fixed": lambda p: chunk_fixed(p, size=chunk_size, overlap=chunk_overlap),
        "recursive": lambda p: chunk_recursive(p, target_size=chunk_size, overlap=chunk_overlap),
        "sentence": lambda p: chunk_sentence(p),  # sentence strategy operates by sentence counts
    }
    
    # comparisons across all three (using standard/configured settings)
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

# ── helper ─────────────────────────────────────────────────────────

def _make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

if __name__ == '__main__':
    chunks = ingest_documents(strategy="recursive")
    console.print(f"\n[bold green]Ingestion complete. {len(chunks)} chunks ready.[/bold green]")
