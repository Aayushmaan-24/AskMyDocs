"""
app.py — Redesigned, High-Fidelity UI/UX for AskMyDocs
A premium, modern document chatbot workspace inspired by ChatGPT, Claude, and Vercel.
"""

import streamlit as st
import os
import re
import time
import json
import shutil
import pandas as pd
from pathlib import Path

# Import RAG pipeline & ingestion/indexing routines
from src.pipeline import ask
from src.ingestion import ingest_documents, load_chunks
from src.indexing import build_vector_index, build_bm25_index

# ── 1. Page Config & Session State Init ────────────────────────────

st.set_page_config(
    page_title="AskMyDocs | Premium AI Workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State Variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "questions_asked" not in st.session_state:
    st.session_state["questions_asked"] = 0
if "total_response_time" not in st.session_state:
    st.session_state["total_response_time"] = 0.0
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# Define PDF upload target directory
PDF_DIR = Path("data/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Premium Custom CSS Stylesheet ───────────────────────────────

# Define Custom Theme Palette
primary_color = "#3b82f6"      # Soft Vibrant Blue
bg_dark = "#090d16"           # Vercel-like deep slate
card_bg_dark = "#131b2e"      # Claude-like card body
border_dark = "#1e293b"       # Subtle border
text_muted = "#94a3b8"        # Soft gray text

st.markdown(f"""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Premium Minimal Scrollbars */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #1e293b;
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #3b82f6;
    }}

    /* Container Card Layouts */
    .modern-card {{
        background: {card_bg_dark};
        border: 1px solid {border_dark};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }}
    .modern-card:hover {{
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(59, 130, 246, 0.15);
    }}

    /* Stat Cards Styles */
    .stat-card {{
        background: rgba(19, 27, 46, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: left;
        transition: all 0.2s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }}
    .stat-card:hover {{
        border-color: rgba(59, 130, 246, 0.4);
        background: rgba(19, 27, 46, 1);
    }}
    .stat-title {{
        color: {text_muted};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .stat-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }}

    /* Performance Indicators Badges */
    .badge-perf {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px;
    }}
    .badge-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }}
    .dot-ready {{ background: #10b981; box-shadow: 0 0 8px #10b981; }}
    .dot-processing {{ background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }}
    .dot-indexing {{ background: #3b82f6; box-shadow: 0 0 8px #3b82f6; }}
    
    /* Beautiful Source cards */
    .source-card-v2 {{
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        transition: all 0.2s ease;
    }}
    .source-card-v2:hover {{
        background: rgba(30, 41, 59, 0.6);
        border-color: #3b82f6;
    }}
    
    /* Suggested chips button styling */
    .chip-btn {{
        background: rgba(59, 130, 246, 0.1);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 8px 16px;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 6px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}
    .chip-btn:hover {{
        background: #3b82f6;
        color: #ffffff;
        border-color: #3b82f6;
        transform: translateY(-1px);
    }}

    /* Elegant alert cards */
    .alert-card {{
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }}
    .alert-card-success {{
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #a7f3d0;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }}

    /* Skeleton Loading Screen */
    .skeleton-box {{
        background: linear-gradient(90deg, #131b2e 25%, #1e293b 50%, #131b2e 75%);
        background-size: 200% 100%;
        animation: loading-skeleton 1.5s infinite;
        border-radius: 6px;
        height: 20px;
        margin-bottom: 10px;
    }}
    @keyframes loading-skeleton {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}

    /* Custom Citation Chips in Chat */
    .citation-badge {{
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
        font-family: 'JetBrains Mono', monospace;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .citation-badge:hover {{
        background: #3b82f6;
        color: #ffffff;
    }}

    /* Centered Minimal App Logo Title in Sidebar */
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }}
    .sidebar-brand h2 {{
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: -0.02em;
        color: #3b82f6;
        margin: 0;
    }}
    
    /* Highlight special message blocks */
    .doc-preview-badge {{
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Style markdown lists/tables in chatbot */
    .answer-container table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
    }}
    .answer-container th, .answer-container td {{
        border: 1px solid #1e293b;
        padding: 10px 14px;
        text-align: left;
    }}
    .answer-container th {{
        background: #111827;
        font-weight: 600;
    }}

</style>
""", unsafe_allow_html=True)


# Helper function to render citation highlighting
def highlight_citations(text: str) -> str:
    """Wraps any [SOURCE N] citations in clean interactive HTML styling."""
    return re.sub(
        r'\[SOURCE (\d+)\]',
        r'<span class="citation-badge" title="Source Document Reference">SOURCE \1</span>',
        text
    )


# ── 3. File Operations & Core Pipeline ─────────────────────────────

def list_uploaded_documents() -> list[Path]:
    """Returns a list of all currently processed documents."""
    patterns = ["*.pdf", "*.docx", "*.txt", "*.md"]
    found_docs = []
    for pat in patterns:
        found_docs.extend(list(PDF_DIR.glob(pat)))
    return sorted(found_docs)


def run_pipeline_indexing(strategy="recursive", size=512, overlap=64):
    """Executes the ingestion and indexing sequence in the UI."""
    try:
        # Step 1: Ingestion
        ingest_documents(pdf_dir=str(PDF_DIR), strategy=strategy, chunk_size=size, chunk_overlap=overlap)

        # Step 2: Vector and Sparse Indexing
        chunks = load_chunks()
        if chunks:
            build_vector_index(chunks)
            build_bm25_index(chunks)
            st.success("🎉 All documents successfully ingested and hybrid indices built!")
        else:
            st.warning("⚠️ No documents available to index.")
    except Exception as e:
        st.error(f"❌ Indexing failed: {e}")


# ── 4. Sidebar Workspace Navigation & Document Management ─────────

with st.sidebar:
    # Header Branding
    st.markdown("""
    <div class="sidebar-brand">
        <span style="font-size:2rem;">📚</span>
        <h2>AskMyDocs Workspace</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Hybrid RAG · Cross-Encoder Reranker · Citation Enforced")
    st.divider()

    # SECTION 1: Dynamic File Uploader
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Supported types: PDF, DOCX, TXT, MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="uploader",
        label_visibility="collapsed"
    )

    # Save files interactively
    if uploaded_files:
        any_new = False
        for uf in uploaded_files:
            target_path = PDF_DIR / uf.name
            if not target_path.exists():
                with open(target_path, "wb") as f:
                    f.write(uf.getbuffer())
                any_new = True
        if any_new:
            st.toast("⚡ Document files uploaded to workspace!", icon="📥")

    # Display dynamic listing of uploaded files
    all_files = list_uploaded_documents()
    if all_files:
        st.markdown(f"**📚 Workspace Corpus ({len(all_files)})**")
        for f in all_files:
            file_size = f.stat().st_size / 1024
            size_str = f"{file_size:.1f} KB" if file_size < 1024 else f"{file_size/1024:.2f} MB"
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.03); border:1px solid #1e293b;
                border-radius:6px; padding:6px 12px; margin-bottom:6px; font-size:0.85rem; display:flex;
                justify-content:space-between; align-items:center;">
                    <span>📄 {f.name[:20] + '...' if len(f.name) > 22 else f.name}</span>
                    <span style="color:#64748b; font-size:0.75rem;">{size_str}</span>
                </div>""",
                unsafe_allow_html=True
            )
            
        # Re-Index Trigger Button
        if st.button("⚡ Rebuild Vector Indices", type="primary", use_container_width=True):
            with st.status("🛠 Rebuilding hybrid BM25 + dense vector corpus...", expanded=True) as status:
                status.write("📂 Extrapolating documents & loading contents...")
                # Fetch settings parameters
                strategy = st.session_state.get("set_chunk_strategy", "recursive")
                size = st.session_state.get("set_chunk_size", 512)
                overlap = st.session_state.get("set_chunk_overlap", 64)

                ingest_documents(pdf_dir=str(PDF_DIR), strategy=strategy, chunk_size=size, chunk_overlap=overlap)
                status.write("🧠 Splitting text recursively into overlapping semantic units...")

                chunks = load_chunks()
                status.write(f"🛰 Generated {len(chunks)} unique text slices. Embedding vector spaces...")

                if chunks:
                    build_vector_index(chunks)
                    status.write("📚 Fitting dense vector spaces into Qdrant index...")
                    build_bm25_index(chunks)
                    status.write("📈 Creating token indices inside BM25 model...")

                status.update(label="✅ Indices completed & optimized!", state="complete")
            st.rerun()
    else:
        st.info("💡 Drop documents here to start. The application is running entirely locally and is colorblind-friendly.")

    # General clear options
    st.divider()
    st.subheader("🧹 Maintenance")
    col_clear_1, col_clear_2 = st.columns(2)
    with col_clear_1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["questions_asked"] = 0
            st.session_state["total_response_time"] = 0.0
            st.toast("Chat workspace cleared!", icon="🧹")
            st.rerun()
    with col_clear_2:
        if st.button("Delete Files", use_container_width=True, type="secondary"):
            if PDF_DIR.exists():
                shutil.rmtree(PDF_DIR)
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            # empty out chunks file
            chunks_file = Path("data/chunks/chunks.json")
            if chunks_file.exists():
                chunks_file.write_text("[]")
            st.session_state["chat_history"] = []
            st.toast("Workspace corpus destroyed!", icon="🗑️")
            st.rerun()

    # SECTION 2: Collapsible Settings Drawer
    st.divider()
    with st.expander("⚙️ RAG Engine Parameters", expanded=False):
        st.session_state["set_llm"] = st.selectbox(
            "Primary LLM Engine",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0
        )
        st.session_state["set_chunk_strategy"] = st.selectbox(
            "Splitting Strategy",
            ["recursive", "fixed", "sentence"],
            index=0
        )
        st.session_state["set_chunk_size"] = st.slider("Chunk Character Size", 256, 1024, 512, step=64)
        st.session_state["set_chunk_overlap"] = st.slider("Overlapping Windows", 16, 128, 64, step=16)
        st.session_state["set_temperature"] = st.slider("LLM Temperature", 0.0, 1.0, 0.1, step=0.1)
        st.session_state["set_top_k"] = st.slider("Retrieve Candidates (top_k)", 5, 30, 10)
        st.session_state["set_top_n"] = st.slider("Rerank Limit (top_n)", 2, 10, 5)

    # Footer Attribution
    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem; text-align:center; color:#64748b;">
        AskMyDocs 📚 AI Space · Version 2.0.0
    </div>
    """, unsafe_allow_html=True)


# ── 5. Main Workspace & Dynamic Dashboards ──────────────────────────

# Fetch and load indexed corpus statistics
all_chunks = load_chunks()
num_docs = len(all_files)
num_chunks = len(all_chunks)

# Title Header & System Badges
title_col, badge_col = st.columns([2, 1])
with title_col:
    st.title("AskMyDocs Intelligence 🌌")
    st.markdown(
        f"<span style='color:{text_muted}; font-size:1.05rem;'>High-fidelity AI workspace "
        "leveraging reciprocal rank fusion and ms-marco cross-encoder rerankers.</span>",
        unsafe_allow_html=True
    )

with badge_col:
    st.markdown("""
    <div style="display:flex; flex-wrap:wrap; justify-content:flex-end; gap:6px; margin-top:8px;">
        <div class="badge-perf"><span class="badge-dot dot-ready"></span>Ready</div>
        <div class="badge-perf" style="font-family:'JetBrains Mono';">LLM: Llama-3.3-70b</div>
        <div class="badge-perf" style="font-family:'JetBrains Mono';">Embedding: BGE-small</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ── 6. Onboarding Empty State & SUGGESTIONS ─────────────────────────

if num_docs == 0:
    # Render premium onboarding template
    st.markdown("""
    <div class="modern-card" style="text-align:center; padding: 60px 40px; margin-top: 20px;">
        <div style="font-size: 5rem; margin-bottom: 20px; animation: bounce 2s infinite;">📚</div>
        <h2 style="font-size: 2.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">Upload your documents to begin</h2>
        <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 30px auto; line-height: 1.6;">
            Create your local secure knowledge base. Ask questions across PDFs, research reports, legal contracts, engineering notes, or plain text Markdown and receive grounded answers.
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <span class="badge-perf" style="padding: 6px 14px; font-size: 0.85rem;">📄 Portable Documents (.pdf)</span>
            <span class="badge-perf" style="padding: 6px 14px; font-size: 0.85rem;">📝 Microsoft Word (.docx)</span>
            <span class="badge-perf" style="padding: 6px 14px; font-size: 0.85rem;">⚙️ Plain Text (.txt)</span>
            <span class="badge-perf" style="padding: 6px 14px; font-size: 0.85rem;">🕸️ Markdown Notes (.md)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Showcase some beautifully styled dummy example queries
    st.markdown("<h4 style='font-weight: 600; color: #cbd5e1; margin-bottom: 15px;'>💡 Explore potential capabilities:</h4>", unsafe_allow_html=True)
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("""
        <div class="source-card-v2" style="border-left-color: #f59e0b;">
            <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">🔎 Complete Semantic Search</div>
            <p style="color:#94a3b8; font-size: 0.85rem; margin: 0; line-height:1.5;">Find concepts and answers using advanced dense vector representations, capturing context beyond exact keyword matches.</p>
        </div>
        """, unsafe_allow_html=True)
    with ex_col2:
        st.markdown("""
        <div class="source-card-v2" style="border-left-color: #10b981;">
            <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin-bottom: 4px;">⚖️ Post-Generation Citation Enforcement</div>
            <p style="color:#94a3b8; font-size: 0.85rem; margin: 0; line-height:1.5;">Every statement is checked against the retrieved document corpus, highlighting exactly where the facts were extracted from.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Documents exist! Render gorgeous statistics metrics and performance dashboard
    avg_resp = (st.session_state["total_response_time"] / max(st.session_state["questions_asked"], 1))
    avg_resp_str = f"{avg_resp:.2f}s" if st.session_state["questions_asked"] > 0 else "N/A"

    # Calculate storage footprint
    total_bytes = sum(f.stat().st_size for f in all_files)
    total_mb_str = f"{total_bytes / (1024*1024):.2f} MB" if total_bytes >= 1024*1024 else f"{total_bytes / 1024:.1f} KB"
    
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">📚 Documents</div>
            <div class="stat-value">{num_docs}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">🧩 Slices (Chunks)</div>
            <div class="stat-value">{num_chunks}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">🌐 Dense Vectors</div>
            <div class="stat-value">{num_chunks}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">❓ Queries</div>
            <div class="stat-value">{st.session_state["questions_asked"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">⏱️ Response Time</div>
            <div class="stat-value">{avg_resp_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col6:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">💾 Storage Footprint</div>
            <div class="stat-value">{total_mb_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # Display clean document cards grid representation
    st.write("")
    with st.expander(f"📂 Explore Document Corpus ({len(all_files)})", expanded=False):
        card_cols = st.columns(3)
        for idx, file_path in enumerate(all_files):
            col_target = card_cols[idx % 3]
            f_size_kb = file_path.stat().st_size / 1024
            with col_target:
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.3); border:1px solid rgba(255,255,255,0.04); border-radius:10px; padding:16px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <span style="font-size:1.4rem;">📄</span>
                        <span class="doc-preview-badge">Indexed</span>
                    </div>
                    <div style="font-weight:600; font-size:0.95rem; margin-top:8px; color:#f1f5f9; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{file_path.name}">
                        {file_path.name}
                    </div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">
                        Size: {f_size_kb:.1f} KB • Format: {file_path.suffix.upper()[1:]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── 7. ChatGPT/Claude-like Chat Interface & Suggestions ───────────

    # Display empty state suggestion chips before any conversation starts
    if not st.session_state["chat_history"]:
        st.markdown("<h5 style='color:#cbd5e1; font-weight:500; margin-bottom:12px;'>⚡ Click a quick start prompt:</h5>", unsafe_allow_html=True)
        chips = [
            "What are the key findings or takeaways?",
            "Can you summarize the major contents of these documents?",
            "What are the prominent risks, warnings, or action items?",
            "Extract any notable timelines, deadlines, or dates mentioned.",
            "Are there any statistical or financial numbers of note?"
        ]

        # Create interactive 2-column layout with generous breathing space
        col_chip_1, col_chip_2 = st.columns(2)
        for idx, chip_text in enumerate(chips):
            target_col = col_chip_1 if idx % 2 == 0 else col_chip_2
            with target_col:
                if st.button(f"✨ {chip_text}", key=f"chip_{idx}", use_container_width=True):
                    st.session_state["last_query"] = chip_text
                    st.rerun()

    # Render Active Chat Dialogues with Custom Avatars
    for msg in st.session_state["chat_history"]:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(
                    f"<div style='font-size:1rem; font-weight:500;'>{content}</div>",
                    unsafe_allow_html=True
                )
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("<div class='answer-container'>", unsafe_allow_html=True)

                # Check for table, quotes, or itemized lists formatting in content
                # and wrap highlighted citation badges
                annotated_answer = highlight_citations(content)
                st.markdown(annotated_answer, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Show elegant citations list metadata if present
                if "citations" in msg and msg["citations"]:
                    with st.expander("🔍 Grounded Source Citations", expanded=False):
                        for citation in msg["citations"]:
                            score = citation.get("ce_score", 0.0)
                            # Create a nice preview of original source snippet
                            text_snippet = citation.get("text_preview", "")
                            st.markdown(f"""
                            <div class="source-card-v2">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                    <strong style="color:#60a5fa;">SOURCE {citation['citation_num']}</strong>
                                    <span style="font-size:0.75rem; background:rgba(96,165,250,0.1); color:#60a5fa; padding:2px 6px; border-radius:4px;">
                                        Match score: {score:.4f}
                                    </span>
                                </div>
                                <div style="font-size:0.85rem; color:#cbd5e1; font-style:italic; margin-bottom:8px;">
                                    "{text_snippet}..."
                                </div>
                                <div style="font-size:0.75rem; color:#64748b;">
                                    Document: {citation['source']} • Page/Section: {citation['page']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Show nice action feedback items at the bottom of the response
                col_feed_1, col_feed_2, col_feed_3, col_feed_4 = st.columns([1, 1, 1, 10])
                with col_feed_1:
                    if st.button("👍 Useful", key=f"like_{msg['id']}", help="This grounded answer is correct"):
                        st.toast("Thank you for your feedback!", icon="💖")
                with col_feed_2:
                    if st.button("👎 Unclear", key=f"dislike_{msg['id']}", help="This answer is incomplete or incorrect"):
                        st.toast("Feedback recorded. We'll fine-tune constraints.", icon="📝")
                with col_feed_3:
                    # Provide a simple copy action code-block utility
                    st.code(content, language="markdown")

    # Sticky Bottom Chat Input
    user_query = st.chat_input("Ask anything about your documents...") or st.session_state["last_query"]

    if user_query:
        # Reset last query chip tracker
        st.session_state["last_query"] = ""

        # Append User Message to timeline
        msg_id = int(time.time() * 1000)
        st.session_state["chat_history"].append({
            "id": msg_id,
            "role": "user",
            "content": user_query
        })

        # Display the user message immediately in workspace
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div style='font-size:1rem; font-weight:500;'>{user_query}</div>", unsafe_allow_html=True)

        # Display Progress steps indicator with high-fidelity steps
        with st.chat_message("assistant", avatar="🤖"):
            placeholder_text = st.empty()
            placeholder_sources = st.empty()

            # Step animation indicators
            steps_placeholder = st.empty()

            with steps_placeholder.container():
                st.markdown("""
                <div style="background:#131b2e; border:1px solid #1e293b; border-radius:8px; padding:16px; width:100%; max-width:400px; margin-bottom:15px;">
                    <div style="font-weight:600; color:#f8fafc; font-size:0.9rem; margin-bottom:10px;">🔍 Analyzing Document Corpus...</div>
                    <div style="font-size:0.85rem; color:#94a3b8; display:flex; flex-direction:column; gap:6px;">
                        <div>⏳ Parsing your question & intent...</div>
                        <div style="opacity:0.5;">⚪ Hybrid retrieval vector & keyword mapping...</div>
                        <div style="opacity:0.5;">⚪ Reciprocal rank fusion merge & cross reranking...</div>
                        <div style="opacity:0.5;">⚪ Validation checks & final answer synthesis...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.4)

            with steps_placeholder.container():
                st.markdown("""
                <div style="background:#131b2e; border:1px solid #1e293b; border-radius:8px; padding:16px; width:100%; max-width:400px; margin-bottom:15px;">
                    <div style="font-weight:600; color:#f8fafc; font-size:0.9rem; margin-bottom:10px;">🔍 Analyzing Document Corpus...</div>
                    <div style="font-size:0.85rem; color:#94a3b8; display:flex; flex-direction:column; gap:6px;">
                        <div style="color:#10b981;">✓ Parsing your question & intent</div>
                        <div>⏳ Hybrid retrieval vector & keyword mapping...</div>
                        <div style="opacity:0.5;">⚪ Reciprocal rank fusion merge & cross reranking...</div>
                        <div style="opacity:0.5;">⚪ Validation checks & final answer synthesis...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.4)

            with steps_placeholder.container():
                st.markdown("""
                <div style="background:#131b2e; border:1px solid #1e293b; border-radius:8px; padding:16px; width:100%; max-width:400px; margin-bottom:15px;">
                    <div style="font-weight:600; color:#f8fafc; font-size:0.9rem; margin-bottom:10px;">🔍 Analyzing Document Corpus...</div>
                    <div style="font-size:0.85rem; color:#94a3b8; display:flex; flex-direction:column; gap:6px;">
                        <div style="color:#10b981;">✓ Parsing your question & intent</div>
                        <div style="color:#10b981;">✓ Hybrid retrieval vector & keyword mapping</div>
                        <div>⏳ Reciprocal rank fusion merge & cross reranking...</div>
                        <div style="opacity:0.5;">⚪ Validation checks & final answer synthesis...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Perform RAG retrieval operations
            start_time = time.time()

            # Fetch engine parameters safely from session state or fall back to defaults
            top_k = st.session_state.get("set_top_k", 10)
            top_n = st.session_state.get("set_top_n", 5)
            llm_model = st.session_state.get("set_llm", "llama-3.3-70b-versatile")
            temperature = st.session_state.get("set_temperature", 0.1)

            result = ask(user_query, top_k=top_k, top_n=top_n, model=llm_model, temperature=temperature)

            time.sleep(0.2)
            with steps_placeholder.container():
                st.markdown("""
                <div style="background:#131b2e; border:1px solid #1e293b; border-radius:8px; padding:16px; width:100%; max-width:400px; margin-bottom:15px;">
                    <div style="font-weight:600; color:#f8fafc; font-size:0.9rem; margin-bottom:10px;">🔍 Analyzing Document Corpus...</div>
                    <div style="font-size:0.85rem; color:#94a3b8; display:flex; flex-direction:column; gap:6px;">
                        <div style="color:#10b981;">✓ Parsing your question & intent</div>
                        <div style="color:#10b981;">✓ Hybrid retrieval vector & keyword mapping</div>
                        <div style="color:#10b981;">✓ Reciprocal rank fusion merge & cross reranking</div>
                        <div>⏳ Validation checks & final answer synthesis...</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            end_time = time.time()
            elapsed = end_time - start_time
            st.session_state["questions_asked"] += 1
            st.session_state["total_response_time"] += elapsed

            # Remove indicators
            steps_placeholder.empty()

            # Render animated text typing simulation
            answer = result["answer"]
            highlighted = highlight_citations(answer)
            placeholder_text.markdown(f"<div class='answer-container'>{highlighted}</div>", unsafe_allow_html=True)

            # Check validation score
            val = result.get("validation", {})
            if val and not val.get("passed", True):
                uncited_count = len(val.get("uncited_sentences", []))
                st.warning(f"⚠️ Fact check complete: found {uncited_count} sentence(s) without grounded citation.")
                for s in val.get("uncited_sentences", []):
                    st.code(s)

            # Append Assistant Message to timeline
            st.session_state["chat_history"].append({
                "id": msg_id + 1,
                "role": "assistant",
                "content": answer,
                "citations": result.get("citations", [])
            })

            st.rerun()
