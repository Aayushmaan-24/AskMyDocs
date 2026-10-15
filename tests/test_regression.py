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

# ── Fixture: run N queries and return metrics ──────────────────────

@pytest.fixture(scope="module")
def fresh_metrics():
    """Run N fresh traced queries and compute metrics."""
    queries = TEST_QUERIES[:N_QUERIES]
    print(f"\nRunning {len(queries)} traced queries for regression check...")
    for q in queries:
        result = traced_ask(q)
        print(f"  ✓ {q[:50]} — {result['trace']['total_ms']:.0f}ms")
    return compute_metrics(limit=N_QUERIES)

# ── Tests ──────────────────────────────────────────────────────────

def test_citation_rate_not_regressed(fresh_metrics):
    """Citation rate must not drop more than 5% below baseline."""
    q = fresh_metrics.get("quality", {})
    rate = q.get("recent_citation_rate", q.get("avg_citation_rate", 0))
    threshold = BASELINE["citation_rate"] - REGRESSION_THRESHOLD
    assert rate >= threshold, (
        f"Citation rate {rate:.2%} is below threshold {threshold:.2%}. "
        f"Baseline: {BASELINE['citation_rate']:.2%}, "
        f"Allowed drift: {REGRESSION_THRESHOLD:.2%}"
    )
    
def test_p95_latency_reasonable(fresh_metrics):
    """p95 end-to-end latency must be under 120 seconds."""
    lat = fresh_metrics.get("latency", {})
    p95 = lat.get("total_p95", 0)
    assert p95 < 120_000, (
        f"p95 latency {p95:.0f}ms exceeds 120s threshold. "
        f"Reranking bottleneck may have worsened."
    )

def test_reranking_under_threshold(fresh_metrics):
    """Reranking step p95 must stay under 90 seconds."""
    lat   = fresh_metrics.get("latency", {})
    steps = lat.get("steps", {})
    rerank_p95 = steps.get("reranking", {}).get("p95", 0)
    assert rerank_p95 < 90_000, (
        f"Reranking p95 {rerank_p95:.0f}ms exceeds 90s. "
        f"Cross-encoder may be degrading on this hardware."
    )
    
def test_cost_per_request_reasonable(fresh_metrics):
    """Cost per request must stay under $0.01."""
    cost = fresh_metrics.get("cost", {})
    avg  = cost.get("avg_per_request", 0)
    assert avg < 0.01, (
        f"Avg cost per request ${avg:.6f} exceeds $0.01 threshold. "
        f"Check prompt length or model pricing."
    )
    
def test_all_steps_present(fresh_metrics):
    """All 6 pipeline steps must be traced on every request."""
    lat      = fresh_metrics.get("latency", {})
    steps    = set(lat.get("steps", {}).keys())
    expected = {
        "bm25_retrieval", "vector_retrieval", "rrf_fusion",
        "reranking", "llm_generation", "citation_validation"
    }
    missing = expected - steps
    assert not missing, f"Missing trace steps: {missing}"