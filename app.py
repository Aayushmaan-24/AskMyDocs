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

