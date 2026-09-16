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