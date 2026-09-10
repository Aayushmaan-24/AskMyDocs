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
        
    
    