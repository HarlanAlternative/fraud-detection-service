"""Prometheus instrumentation for the scoring service.

These series back the Grafana dashboard and the drift alerts: decision mix, scoring latency
(for the P95 < 100 ms SLA), and the live prediction-score distribution (whose shift is one of
the drift signals).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

SCORES_TOTAL = Counter(
    "fraud_scores_total", "Scored transactions by decision", ["decision"]
)
REQUESTS_TOTAL = Counter(
    "fraud_requests_total", "API requests", ["endpoint", "status"]
)
SCORING_LATENCY = Histogram(
    "fraud_scoring_latency_seconds",
    "End-to-end scoring latency (incl. SHAP)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
SCORE_DISTRIBUTION = Histogram(
    "fraud_score_distribution",
    "Distribution of emitted fraud scores (for prediction-drift monitoring)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_LOADED = Gauge("fraud_model_loaded", "1 when a model is loaded and ready")


def observe_result(result: dict) -> None:
    """Update score/decision series from one scoring result."""
    SCORES_TOTAL.labels(decision=result["fraud_decision"]).inc()
    SCORE_DISTRIBUTION.observe(result["fraud_score"])
