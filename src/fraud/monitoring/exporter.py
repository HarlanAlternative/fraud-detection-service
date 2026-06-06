"""Drift-monitor exporter.

Periodically compares recent live traffic (from the Postgres inference log) against the
training *reference* distribution and publishes Prometheus gauges that the alert rules in
``monitoring/prometheus/alerts.yml`` watch:

* ``fraud_feature_psi{feature}`` — per-feature PSI
* ``fraud_score_psi``           — prediction-score distribution PSI
* ``fraud_backfill_recall``     — recall on recently backfilled labels

Run as a service: ``python -m fraud.monitoring.exporter`` (serves :9100).
One-shot offline check: ``python -m fraud.monitoring.exporter --demo``.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from prometheus_client import Gauge, start_http_server

from fraud.config import get_settings
from fraud.data.schema import CURATED_CATEGORICAL, CURATED_FEATURES
from fraud.monitoring.drift import classify_psi, feature_psi, psi, rolling_recall

FEATURE_PSI = Gauge("fraud_feature_psi", "Per-feature PSI vs training reference", ["feature"])
SCORE_PSI = Gauge("fraud_score_psi", "Prediction-score distribution PSI vs reference")
BACKFILL_RECALL = Gauge("fraud_backfill_recall", "Rolling recall on backfilled labels")

_CAT = set(CURATED_CATEGORICAL)


def load_reference() -> tuple[pd.DataFrame, pd.Series]:
    """Load the saved training reference (features + scores); fall back to synthetic."""
    settings = get_settings()
    ref_path = settings.models_dir / "reference.parquet"
    if ref_path.exists():
        ref = pd.read_parquet(ref_path)
        return ref[CURATED_FEATURES], ref["fraud_score"]
    # Fallback: rebuild a reference from data so the exporter still runs.
    from fraud.train.dataset import load_raw, time_split

    splits = time_split(load_raw(20_000))
    return splits.X_train, pd.Series([], dtype=float)


def compute_once(ref_X: pd.DataFrame, ref_scores: pd.Series, logger) -> dict:
    """Compute drift gauges from the most recent logged traffic. Returns a summary dict."""
    rows = logger.recent_labeled(limit=5000) if logger and logger.enabled else []
    summary: dict = {"n_current": 0}

    # Pull a current window of features/scores from the inference log.
    current = logger.engine and pd.read_sql(
        "SELECT features, fraud_score, fraud_decision, label FROM inference_log "
        "ORDER BY scored_at DESC LIMIT 5000",
        logger.engine,
    ) if (logger and logger.enabled) else None

    if current is not None and len(current):
        feats = pd.json_normalize(current["features"])
        psis = feature_psi(ref_X, feats, CURATED_FEATURES, _CAT)
        for f, v in psis.items():
            FEATURE_PSI.labels(feature=f).set(v)
        if len(ref_scores):
            sp = psi(ref_scores.to_numpy(), current["fraud_score"].to_numpy())
            SCORE_PSI.set(sp)
            summary["score_psi"] = sp
        summary["n_current"] = len(current)
        summary["max_feature_psi"] = max(psis.values()) if psis else 0.0

    rec = rolling_recall(rows)
    if rec == rec:  # not NaN
        BACKFILL_RECALL.set(rec)
        summary["backfill_recall"] = rec
    return summary


def _demo() -> int:
    """Offline check: reference vs a synthetic drifted window (no DB / HTTP)."""
    from fraud.data.synthetic import make_ieee_like
    from fraud.features.engineering import build_training_frame

    ref = build_training_frame(make_ieee_like(20_000, seed=1))[0]
    cur = build_training_frame(make_ieee_like(20_000, seed=9, drift=True))[0]
    psis = feature_psi(ref, cur, CURATED_FEATURES, _CAT)
    print("feature PSI (reference vs drifted window):")
    for f, v in sorted(psis.items(), key=lambda x: -x[1])[:8]:
        print(f"  {f:24s} {v:.4f}  {classify_psi(v)}")
    flagged = [f for f, v in psis.items() if v > 0.2]
    print(f"\nfeatures flagged as significant drift (PSI>0.2): {flagged}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fraud drift-monitor exporter.")
    parser.add_argument("--demo", action="store_true", help="one-shot offline drift check")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args(argv)

    if args.demo:
        return _demo()

    from fraud.serving.db import InferenceLogger

    ref_X, ref_scores = load_reference()
    logger = InferenceLogger()
    start_http_server(args.port)
    print(f"[drift-monitor] serving :{args.port}, refreshing every {args.interval}s")
    while True:
        try:
            summary = compute_once(ref_X, ref_scores, logger)
            print(f"[drift-monitor] {summary}")
        except Exception as exc:  # noqa: BLE001 - never crash the monitor
            print(f"[drift-monitor] compute error: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
