"""Drift metrics: Population Stability Index (PSI) and rolling backfill recall.

PSI compares a *reference* distribution (the training data the model was fit on) with a
*current* window of live traffic. The industry rule of thumb: PSI < 0.1 = stable,
0.1-0.2 = moderate shift, > 0.2 = significant drift (retrain). We track PSI on each feature,
on the prediction-score distribution, and recall on whatever labels have backfilled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two 1-D numeric samples.

    Bucket edges come from the reference quantiles (so each reference bucket holds ~equal
    mass); we then compare the fraction of ``current`` falling in each bucket.
    """
    reference = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:  # constant reference -> no meaningful PSI
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature (compares category frequencies)."""
    ref = reference.astype("string").fillna("missing").value_counts(normalize=True)
    cur = current.astype("string").fillna("missing").value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    ref_p = np.clip(ref.reindex(cats, fill_value=0).to_numpy(), EPS, None)
    cur_p = np.clip(cur.reindex(cats, fill_value=0).to_numpy(), EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def feature_psi(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], categorical: set[str]
) -> dict[str, float]:
    """Per-feature PSI across a reference and current frame."""
    out: dict[str, float] = {}
    for f in features:
        if f not in reference.columns or f not in current.columns:
            continue
        out[f] = (
            categorical_psi(reference[f], current[f])
            if f in categorical
            else psi(reference[f].to_numpy(), current[f].to_numpy())
        )
    return out


def rolling_recall(labeled_rows: list[dict], max_fpr_decisions=("decline", "review")) -> float:
    """Recall on backfilled-label rows: of true frauds, how many were declined/reviewed."""
    frauds = [r for r in labeled_rows if int(r.get("label", 0)) == 1]
    if not frauds:
        return float("nan")
    caught = sum(1 for r in frauds if r.get("fraud_decision") in max_fpr_decisions)
    return caught / len(frauds)


def classify_psi(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.2:
        return "moderate"
    return "significant"
