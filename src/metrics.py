"""
metrics.py — Compute p50/p95 latency, cost totals, quality drift
Reads from SQLite traces written by observability.py
"""

import json
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from rich.console import Console
from rich.table import Table

from src.observability import load_traces, DB_PATH

console = Console()

# baseline scores for validation
BASELINE = {
    "faithfulness":    0.80,
    "citation_rate":   1.00,
    "answer_relevancy": 0.98,
}

REGRESSION_THRESHOLD = 0.05  # 5% drop triggers alert

# ── 1. Percentile helper ───────────────────────────────────────────

def percentile(data : list[float], p: int) -> float:
    
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return round(sorted_data[idx], 1)

# ── 2. Latency metrics ─────────────────────────────────────────────

def compute_latency_metrics(traces: list[dict]) -> dict:
    """p50/p95 for total latency and per step latency"""
    total_times = [t["total_ms"] for t in traces if t.get("total_ms")]
    
    step_times = defaultdict(list)
    for t in traces:
        steps = t.get("step_durations", {})
        for step , ms in steps.items():
            step_times[step].append(ms)
            
    return {
        "total_p50" : percentile(total_times, 50),
        "total_p95" : percentile(total_times, 95),
        "total_avg" : round(sum(total_times) / len(total_times), 1) if total_times else 0.0,
        "steps" : {
            step : {
                "p50" : percentile(times, 50),
                "p95" : percentile(times, 95),
            }
            for step, times in step_times.items()
        }
    }
    
# ── 3. Cost metrics ────────────────────────────────────────────────

def compute_cost_metrics(traces: listp[dict]) -> dict:
    """Total cost, cost per request, daily projection."""
    
    costs = [t["cost_usd"] for t in traces if t.get("cost_usd")]
    if not costs:
        return {}
    
    total = round(sum(costs), 4)
    avg = round(total / len(costs), 4)
    
    if len(traces) >= 2:
        first = datetime.fromisoformat(traces[-1]["timestamp"])
        last = datetime.fromisoformat(traces[0]["timestamp"])
        hours = max((last - first).total_seconds()/ 3600, 0.001)
        rph = len(traces) / hours
        daily = round(rph * 24 * avg, 4)
        
    else:
        daily = 0.0
        
    return {
        "total_cost" : total,
        "avg_cost" : avg,
        "daily_projection" : daily,
        "total_requests" : len(costs),
        "total_tokens" : sum((t.get("prompt_tokens", 0) or 0) + (t.get("completion_tokens", 0) or 0) for t in traces),
    }
    
# ── 4. Quality metrics ─────────────────────────────────────────────

def compute_quality_metrics(traces: list[dict]) -> dict:
    """Rolling citation rate + regression detection."""
    
    citation_rates = [
        t["citation_rate"] for t in traces if t.get("citation_rate") is not None
    ]
    
    avg_citation = round(sum(citation_rates) / len(citation_rates), 3) if citation_rates else 0.0
    
    recent = citation_rates[:10]  # last 10 traces
    recent_avg = round(sum(recent) / len(recent), 3) if recent else 0.0
    drift = round(abs(recent_avg - BASELINE["citation_rate"]), 3)
    
    return {
        "avg_citation_rate" : avg_citation,
        "recent_avg_citation_rate" : recent_avg,
        "drift_from_baseline" : drift,
        "regression_alert" : drift > REGRESSION_THRESHOLD,
    }
    
# ── 5. Full report ─────────────────────────────────────────────────

def compute_metrics(limit: int = 100) -> dict:
    traces = load_traces(limit=limit)
    if not limit:
        return {
            "error" : "No traces found. Run some queries first."
        }
        
    return {
        "trace_count" : len(traces),
        "latency" : compute_latency_metrics(traces),
        "cost" : compute_cost_metrics(traces),
        "quality" : compute_quality_metrics(traces),
    }