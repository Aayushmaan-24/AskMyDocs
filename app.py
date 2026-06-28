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