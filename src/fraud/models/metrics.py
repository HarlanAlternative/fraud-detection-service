"""Evaluation metrics centred on the business constraint: recall at a capped FPR.

The headline number for this project is **recall at false-positive-rate = 2%** — the rate
at which real fraud is caught while only disturbing 2% of legitimate customers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y_true, scores, max_fpr: float = 0.02) -> float:
    """Recall (TPR) at the operating point whose FPR is as high as possible but <= ``max_fpr``."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    allowed = fpr <= max_fpr
    if not allowed.any():
        return 0.0
    return float(tpr[allowed].max())


def threshold_at_fpr(y_true, scores, max_fpr: float = 0.02) -> float:
    """The score threshold that yields the highest recall while keeping FPR <= ``max_fpr``."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    allowed = fpr <= max_fpr
    if not allowed.any():
        return 1.0
    idx_candidates = np.where(allowed)[0]
    best = idx_candidates[np.argmax(tpr[idx_candidates])]
    return float(thresholds[best])


def evaluate(y_true, scores, max_fpr: float = 0.02) -> dict[str, float]:
    """Return the standard metric bundle for one model's scores."""
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        f"recall_at_fpr_{max_fpr:g}": recall_at_fpr(y_true, scores, max_fpr),
    }
