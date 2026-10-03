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

# ── Load data ──────────────────────────────────────────────────────

traces = load_traces(limit = 200)
metrics = compute_metrics()

if not traces:
    st.warning("No traces found. Please run some queries first.")
    st.stop()
    
df = pd.DataFrame(traces)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["total_s"] = df["total_ms"] / 1000
df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)

# parse step durations

steps_df_rows = []

for _, row in df.iterrows():
    steps = row.get("step_durations", {})
    if isinstance(steps, str):
        steps = json.loads(steps)
    for step, ms in steps.items():
        steps_df_rows.append({
            "timestamp" : row["timestamp"],
            "step" : step,
            "ms" : ms,
            "request_id" : row["request_id"],
        })

step_df = pd.DataFrame(steps_df_rows)

# ── Top metrics row ────────────────────────────────────────────────

st.divider()
col1, col2, col3, col4, col5 = st.columns(5)

latency = metrics.get("latency", {})
cost = metrics.get("cost", {})
quality = metrics.get("quality", {})    

with col1:
    st.metric("Total Requests", metrics.get("trace_count", 0))
    
with col2:
    st.metric("p50 latency", f"{latency.get("total_p50", 0)/1000:.1f}s")

with col3:
    st.metric("p95 latency", f"{latency.get("total_p95", 0)/1000:.1f}s")
    
with col4:
    st.metric("Average cost / req", f"${cost.get("avg_per_request", 0):.6f}")
    
with col5:
    cr = quality.get("avg_citation_rate", 0)
    st.metric("Citation Rate", f"{cr:.0%}", delta = f"{cr - BASELINE['citation_rate']:.0%} vs baseline")
    
st.divider()

# ── Row 1: Latency waterfall + step breakdown ──────────────────────

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Step Latency Breakdown (p50)")
    steps_order = ["bm25_retrieval", "vector_retrieval", "rrf_fusion",
                   "reranking", "llm_generation", "citation_validation"]
    
    if not step_df.empty:
        p50_per_step = (
            step_df.groupby("step")["ms"]
            .median().reindex(steps_order).dropna().reset_index()
        )
        p50_per_step.columns = ["step", "p50_ms"]
    
        colors = ["#06b6d4", "#7c3aed", "#10b981", "#ef4444", "#f59e0b", "#6366f1"]
        
        fig = go.Figure(go.Bar(
            x=p50_per_step["p50_ms"],
            y=p50_per_step["step"],
            orientation="h",
            marker_color=colors[:len(p50_per_step)],
            text=[f"{v:.0f}ms" for v in p50_per_step["p50_ms"]],
            textposition="outside",
        ))
        
        fig.update_layout(
            height=300, margin=dict(l=0, r=60, t=20, b=0),
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#e2e8f0", xaxis_title="milliseconds",
            yaxis=dict(categoryorder="array", categoryarray=steps_order[::-1]),
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("End-to-End Latency Over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["timestamp"], y=df["total_s"],
        mode="lines+markers",
        line=dict(color="#7c3aed", width=2),
        marker=dict(size=6),
        name="Total (s)",
    ))
    fig2.add_hline(
        y=df["total_s"].median(),
        line_dash="dash", line_color="#06b6d4",
        annotation_text=f"p50: {df['total_s'].median():.1f}s",
    )
    fig2.update_layout(
        height=300, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", yaxis_title="seconds",
    )
    st.plotly_chart(fig2, use_container_width=True)