"""
app.py — Streamlit UI for Ask My Docs
Citation highlighting, source explorer, retrieval scores
"""

import streamlit as st
import re
from src.pipeline import ask

# ── Page config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📚",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────

st.markdown("""
<style>
    .citation-badge {
        background: #1e3a5f;
        color: #60a5fa;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: monospace;
    }
    
    .source-card {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-left: 3px solid #60a5fa;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.85rem;
    }
    
    .score-good {
        color: #34d399; 
        font-weight: 600;
    }
    
    .score-mid {
        color: #fbbf24; 
        font-weight: 600;
    }
    
    .score-bad {
        color: #f87171; 
        font-weight: 600;
    }
    
    .validation-pass {
        color: #34d399; 
        font-weight: 700;
    }
    
    .validation-fail {
        color: #f87171; 
        font-weight: 700;
    }
    
    .answer-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px 24px;
        line-height: 1.8;
        font-size: 0.95rem;
    }
    
    .metric-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 10px 16px;
        text-align: center;
    }

</style>
            
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────

def highlight_citations(text: str) -> str:
    """Wrap [SOURCE N] tags in styled HTML spans."""
    return re.sub(
        r'\[SOURCE (\d+)\]',
        r'<span class="citation-badge">[SOURCE \1]</span>',
        text
    )
    
def score_color(score: float) -> str:
    """return a CSS class based on the score value."""
    if score > 0:
        return "score-good"
    if score > -5:
        return "score-mid"
    return "score-bad"

# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    
    st.title("📚 Ask My Docs")
    st.caption("RAG · Hybrid Retrieval · Citation Enforced")
    st.divider()
    
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Candidates to retrieve (top_k)", 5, 20, 10)
    top_n = st.slider("Results after reranking (top_n)", 2, 8, 5)
    st.divider()
    
    st.subheader("📖 About")
    st.markdown("""
    **Pipeline:**
    1. BM25 sparse retrieval
    2. Vector dense retrieval
    3. RRF fusion
    4. Cross-encoder reranking
    5. Groq LLM generation
    6. Citation validation
    
    **Model:** `llama-3.3-70b-versatile`
    **Embeddings:** `bge-small-en-v1.5`
    **Reranker:** `ms-marco-MiniLM-L-6-v2`
    """)
    st.divider()
    
    st.caption("Drop PDFs in `data/pdfs/` and re-run ingestion + indexing to update the corpus.")
    
# ── Main ───────────────────────────────────────────────────────────

st.header("Ask My Docs")
st.caption("Every answer is grounded in your documents. No hallucinations — citations enforced.")

# example queries
with st.expander("💡 Example queries", expanded=False):
    examples = [
        "What position is the applicant applying for?",
        "What projects or experience does the applicant mention?",
        "Why does the applicant want to work at Google?",
        "What courses has the applicant completed?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["query_input"] = ex
            
query = st.text_input("Ask a question about your documents:", 
                      placeholder="e.g. What is the main argument of the paper?",
                      key="query_input"
)

run = st.button("Ask", type="primary", disabled=not query)

if run and query:
    with st.spinner("Retrieving · Reranking · Generating..."):
        result = ask(query, top_k=top_k, top_n=top_n)
        
    # ── Answer ──
    st.subheader("Answer")
    highlighted = highlight_citations(result["answer"])
    st.markdown(
        f'<div class="answer-box">{highlighted}</div>',
        unsafe_allow_html=True
    )
    st.divider()
    
    # ── Metrics ──
    v = result['validation']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Citation Rate", f"{v['citation_rate']*100:.0f}%")
    with col2:
        st.metric("Sentences", v["total_sentences"])
    with col3:
        st.metric("Sources Used", len(result["citations"]))
    with col4:
        st.metric("Tokens Used",
                  result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"])
        
    # ── citation status ──
    if v['passed']:
        st.success("✓ Citation validation passed — every sentence is grounded")
    else:
        st.error(f"✗ {len(v['uncited_sentences'])} uncited sentence(s) found")
        for s in v['uncited_sentences']:
            st.code(s)
    st.divider()
    
    # ── Citations ──
    st.subheader("Sources")
    for c in result["citations"]:
        score = c["ce_score"]
        if score > 0:
            score_color_hex = "#34d399"
            score_label = "relevant"
        elif score > -5:
            score_color_hex = "#fbbf24"
            score_label = "moderate"
        else:
            score_color_hex = "#f87171"
            score_label = "low"

        # clean up double spaces from pypdf
        preview = " ".join(c["text_preview"].split())
        with st.container():
            col_badge, col_body = st.columns([1, 8])
            with col_badge:
                st.markdown(
                    f"<div style='background:#1e3a5f;color:#60a5fa;padding:6px 10px;"
                    f"border-radius:6px;font-family:monospace;font-weight:700;"
                    f"text-align:center;margin-top:4px'>S{c['citation_num']}</div>",
                    unsafe_allow_html=True
                )
            with col_body:
                st.markdown(
                    f"**{c['source']}** · page {c['page']} &nbsp;&nbsp;"
                    f"<span style='color:{score_color_hex};font-size:0.8rem'>"
                    f"● ce_score: {score:.4f} ({score_label})</span>",
                    unsafe_allow_html=True
                )
                st.caption(f'"{preview}..."')
            st.divider()
    
    # ── All retrieved chunks (expandable) ──
    
    with st.expander("🔍 All retrieved chunks (before reranking cutoff)", expanded=False):
        for i , chunk in enumerate(result['chunks'], 1):
            with st.container():
                cols = st.columns([1, 6, 1])
                with cols[0]:
                    st.markdown(f"**#{i}**")
                with cols[1]:
                    st.markdown(f"`{chunk['source']}` · page {chunk['page']}")
                    st.caption(chunk["text"][:200])
                with cols[2]:
                    st.markdown(
                        f"<span class='{score_color(chunk.get('ce_score',0))}'>"
                        f"{chunk.get('ce_score', chunk.get('rrf_score', 0)):.4f}</span>",
                        unsafe_allow_html=True
                    )
                st.divider()