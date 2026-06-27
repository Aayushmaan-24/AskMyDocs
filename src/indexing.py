"""
indexing.py — Build Qdrant vector index + BM25 index from chunks
"""

import json
import pickle
from pathlib import Path
import bm25s
import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import(
    Distance,
    PointStruct,
    VectorParams
)
from rich.console import Console
from rich.progress import track

console = Console()

COLLECTION_NAME = "AskMyDocs"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"  # 384-dim, ~130MB, fully local
QDRANT_PATH     = "data/qdrant"
BM25_PATH       = "data/chunks/bm25_index.pkl"

# ── 1. Embedder (singleton) ────────────────────────────────────────

_embedder = None

def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        console.print("[cyan]Loading embedding model (first run downloads ~ 130MB)...[/cyan]")
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder

def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    vectors = list(embedder.embed(texts))
    return [v.tolist() for v in vectors]

