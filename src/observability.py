"""
observability.py — Request tracing + cost tracking for AskMyDocs
Every RAG request is traced with per-step latency and cost.
Storage: SQLite (local, zero setup)
"""

import time
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional

DB_PATH = "data/traces.db"

# Groq pricing (per 1M tokens, as of 2026)
PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "default":                  {"input": 0.59, "output": 0.79},
}

# ── 1. Data structures ─────────────────────────────────────────────

@dataclass
class StepTrace:
    
    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    
    def finish(self, **metadata):
        self.end_time = time.perf_counter()
        self.metadata.update(metadata)
        return self
    
    @property
    def duration_ms(self):
        if self.end_time is None:
            return 0.0
        return round((self.end_time - self.start_time) * 1000, 2)
        
@dataclass
class RequestTrace:
    query : str
    request_id : str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    timestamp : str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = "llama-3.3-70b-versatile"
    steps: list = field(default_factory=list)
    
        # filled after completion
    total_ms:            float = 0.0
    prompt_tokens:       int   = 0
    completion_tokens:   int   = 0
    cost_usd:            float = 0.0
    citation_rate:       float = 0.0
    faithfulness:        float = 0.0
    chunks_retrieved:    int   = 0
    top_ce_score:        float = 0.0
    error:               Optional[str] = None
    
    def add_step(self, name: str) -> StepTrace:
        step = StepTrace(name)
        self.steps.append(step)
        return step
    
    def compute_cost(self):
        pricing = PRICING.get(self.model, PRICING["default"])
        input_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        self.cost_usd = round(input_cost + output_cost, 6)
        
    def step_duration(self) -> dict:
        return {s.name : s.duration_ms for s in self.steps}
    
# ── 2. SQLite storage ──────────────────────────────────────────────

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            request_id        TEXT PRIMARY KEY,
            timestamp         TEXT,
            query             TEXT,
            model             TEXT,
            total_ms          REAL,
            prompt_tokens     INTEGER,
            completion_tokens INTEGER,
            cost_usd          REAL,
            citation_rate     REAL,
            faithfulness      REAL,
            chunks_retrieved  INTEGER,
            top_ce_score      REAL,
            step_durations    TEXT,
            error             TEXT
        )
    """)
    conn.commit()
    conn.close()
    
def save_trace(trace: RequestTrace):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO traces VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        trace.request_id,
        trace.timestamp,
        trace.query,
        trace.model,
        trace.total_ms,
        trace.prompt_tokens,
        trace.completion_tokens,
        trace.cost_usd,
        trace.citation_rate,
        trace.faithfulness,
        trace.chunks_retrieved,
        trace.top_ce_score,
        json.dumps(trace.step_durations()),
        trace.error,
    ))
    conn.commit()
    conn.close()
    
def load_traces(limit: int = 500) -> list[dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM traces ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    
    results = []
    
    for row in rows:
        row_dict = dict(row)
        row_dict["step_durations"] = json.loads(row_dict["step_durations"] or "{}")
        results.append(row_dict)
        
    return results