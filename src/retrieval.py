"""
retrieval.py — Hybrid retrieval: BM25 + Vector via RRF, then cross-encoder reranking
"""

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder
from rich.console import Console
from src.indexing import (
    COLLECTION_NAME,
    embed_texts,
    load_bm25,
    load_vector_client,
)

console = Console()

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

# ── 1. Reranker (singleton) ────────────────────────────────────────

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        console.print("[cyan]Loading cross-encoder reranker...[/cyan]")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

# ── 2. Individual retrievers ───────────────────────────────────────

def retrieve_vector(query: str, top_k: int = 10) -> list[dict]:
    """Dense retrieval from Qdrant."""
    client = load_vector_client()
    query_vec = embed_texts([query])[0]
    results = client.search(
        collection_name = COLLECTION_NAME,
        query_vector = query_vec,
        limit = top_k,
    )
    
    return [
        {
            "chunk_id" : r.payload['chunk_id'],
            "text" : r.payload['text'],
            "source" : r.payload['source'],
            "page" : r.payload['page'],
            "score" : r.score,
            "method" : "vector",
        }
        for r in results
    ]
    
    
def retrieve_bm25(query: str, top_k : int = 10) -> list[dict]:
    """Sparse retrieval via BM25."""
    bm25_index , corpus = load_bm25()
    tokenized_query = bm25s.tokenize([query], stopwords='en')
    results , scores = bm25_index.retrieve(tokenized_query, k = min(top_k, len(corpus)))
    
    hits = []
    for idx, score in zip(results[0], scores[0]):
        chunk = corpus[idx]
        hits.append({
            "chunk_id" : chunk['chunk_id'],
            "text" : chunk['text'],
            "page" : chunk['page'],
            "source" : chunk['source'],
            "score" : float(score),
            "method" : "bm25",
        })
        
    return hits

# ── 3. Reciprocal Rank Fusion ──────────────────────────────────────

def reciprocal_rank_fusion(results_a: list[dict], results_b: list[dict], k: int = 60) -> list[dict]:
    """
    Merge two ranked lists via RRF.
    RRF score = 1/(k + rank_a) + 1/(k + rank_b)
    Higher is better.
    """
    
    scores: dict[str, float] = {}
    docs: dict[str, float] = {}
    
    for rank, doc in enumerate(results_a):
        cid = doc['chunk_id']
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        docs[cid] = doc
        
    for rank, doc in enumerate(results_b):
        cid = doc['chunk_id']
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        docs[cid] = doc
        
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**docs[cid], "rrf_score": round(score, 6), "method" : "hybrid"}
        for cid, score in merged
    ]
    
# ── 4. Cross-encoder reranking ─────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """Score each candidate with cross-encoder, return top_n."""
    if not candidates:
        return []
    
    reranker = get_reranker()
    pairs = [(query, c['text']) for c in candidates]
    ce_scores = reranker.predict(pairs)
    
    for i, doc in enumerate(candidates):
        doc['ce_score'] = round(float(ce_scores[i]), 4)
        
    reranked = sorted(candidates, key=lambda x: x['ce_score'], reverse=True)
    return reranked[:top_n]
