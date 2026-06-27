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

# ── 2. Qdrant vector index ─────────────────────────────────────────

def build_vector_index(chunks: list[dict]) -> QdrantClient:
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)
    
    # Drop + recreate for clean rebuild
    
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        console.print("[yellow]Dropped existing Qdrant collection[/yellow]")
        
    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config=VectorParams(size = 384, distance=Distance.COSINE),
    )
    
    texts = [c["text"] for c in chunks]
    console.print(f"[cyan]Embedding {len(texts)} chunks...[/cyan]")
    vectors = embed_texts(texts)
    
    points = [
        PointStruct(
            id = i,
            vector = vectors[i],
            payload={
                "text": chunks[i]['text'],
                "source": chunks[i]['source'],
                "page": chunks[i]['page'],
                "chunk_id": chunks[i]['chunk_id'],
            },
        )
        for i in track(range(len(chunks)), description="Indexing into Qdrant...")
    ]
    
    client.upsert(collection_name = COLLECTION_NAME, points = points)
    console.print(f"[green]✓ Qdrant: {len(points)} vectors indexed[/green]")
    return client

# ── 3. BM25 index ─────────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> bm25s.BM25:
    texts = [c['text'] for c in chunks]
    
    # tokenize
    tokenized = bm25s.tokenize(texts, stopwords="en")
    
    # build index
    bm25_index = bm25s.BM25()
    bm25_index.index(tokenized)
    
    # save index and corpus
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index":bm25_index, "corpus": chunks}, f)
    
    console.print(f"[green]✓ BM25: {len(texts)} documents indexed → {BM25_PATH}[/green]")
    return bm25_index

# ── 4. Load helpers (used by retrieval.py) ────────────────────────

def load_vector_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)

def load_bm25() -> tuple[bm25s.BM25, list[dict]]:
    with open(BM25_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['index'], data['corpus']