# AskMyDocs 📚

A production-grade RAG system with hybrid retrieval, citation enforcement, and a CI-gated evaluation pipeline.

## Architecture

Query
│
├─► BM25 sparse retrieval
├─► Vector dense retrieval (BGE embeddings)
│
├─► RRF fusion (Reciprocal Rank Fusion)
├─► Cross-encoder reranking (ms-marco-MiniLM)
│
├─► Groq LLM generation (llama-3.3-70b)
├─► Citation enforcement (every sentence cited)
└─► LLM-as-judge eval gate (CI blocks on regression)

## Eval Results (LLM-as-Judge)

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.80 | 0.70 | ✓ PASS |
| Answer Relevancy | 0.98 | 0.70 | ✓ PASS |
| Context Precision | 0.26 | 0.25 | ✓ PASS |
| Context Recall | 0.96 | 0.60 | ✓ PASS |

> Context precision is low with a small corpus (<20 chunks). Improves significantly with more PDFs.

## Chunking Strategy Comparison

| Strategy | Chunks | Avg Length |
|---|---|---|
| Fixed-size | 10 | 470 chars |
| Recursive | 9 | 465 chars |
| Sentence-aware | 6 | 863 chars |

**Chosen:** Recursive — best balance of semantic coherence and chunk density.

## Stack

| Layer | Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, no API cost) |
| Vector DB | Qdrant (persisted locally) |
| Sparse retrieval | BM25S |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq `llama-3.3-70b-versatile` |
| Eval | LLM-as-judge (custom, no RAGAS dependency) |
| CI | GitHub Actions — blocks merge on quality regression |
| UI | Streamlit |

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/Aayushmaan-24/AskMyDocs
cd AskMyDocs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# edit .env → add GROQ_API_KEY=your_key

# 3. Drop PDFs into data/pdfs/, then ingest
python -m src.ingestion
python -m src.indexing

# 4. Run the UI
streamlit run app.py

# 5. Run eval
python -m tests.eval_ragas
```

## CI Gate

Every push to `main` triggers:
1. Full ingestion + indexing on the committed PDF corpus
2. 5-question eval subset (cost-efficient)
3. Pytest threshold checks — merge blocked if any metric regresses

## What makes this production-grade

- **Hybrid retrieval** — BM25 catches exact keywords, vectors catch semantic meaning, RRF merges both without score normalization issues
- **Reranking** — cross-encoder reads query+chunk together (not independently), giving far more accurate relevance scores than embedding similarity alone
- **Citation enforcement** — every sentence in every answer must cite a source chunk. Uncited sentences are flagged. 98% answer relevancy across eval set.
- **CI-gated eval** — quality metrics run on every push. A retrieval change that hurts faithfulness gets caught before it merges.
