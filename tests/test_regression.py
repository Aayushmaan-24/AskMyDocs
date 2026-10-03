"""
test_regression.py — Regression gate for observability metrics
Runs a small set of traced queries and checks quality thresholds.
Blocks CI if citation rate or latency regresses beyond threshold.
Run: pytest tests/test_regression.py -v
"""

import os
import pytest
from src.observability import traced_ask, load_traces
from src.metrics import compute_metrics, BASELINE, REGRESSION_THRESHOLD

# In CI, run fewer queries to save API cost
CI_MODE   = os.getenv("CI", "false").lower() == "true"
N_QUERIES = 3 if CI_MODE else 5

TEST_QUERIES = [
    "What position is the applicant applying for?",
    "Who is the applicant?",
    "Where is the Google office the applicant is applying to?",
    "What courses has the applicant completed outside their curriculum?",
    "Why does the applicant want to work at Google?",
]

