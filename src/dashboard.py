"""
dashboard.py — Observability dashboard for AskMyDocs
Shows latency breakdown, cost tracking, quality drift
Run: streamlit run src/dashboard.py
"""

import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from src.observability import load_traces
from src.metrics import compute_metrics, BASELINE

st.set_page_config(
    page_title="AskMyDocs Observability Dashboard",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; color: #7c3aed; }
  [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────

st.title("📡 AskMyDocs Observability")
st.caption("Latency · Cost · Quality drift · Step breakdown")

