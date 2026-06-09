"""Threshold tuning + the approve / review / decline decision policy.

Banks don't ship a raw probability — they ship a *decision*, and they balance two
constraints that usually conflict: catch as much fraud as possible (recall) while
auto-declining almost no legitimate customers (FPR). We resolve this with **two**
thresholds tuned on a holdout slice:

* ``decline_threshold`` — set at the FPR budget (default 2%): only the highest-confidence
  scores are auto-declined, so few good customers are blocked.
* ``review_threshold`` — set at the target-recall operating point (default 0.90): scores
  between the two thresholds are routed to manual review, so decline+review together reach
  the recall target.

Everything below ``review_threshold`` is auto-approved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve


def threshold_for_recall(y_true, scores, target_recall: float = 0.90) -> float:
    """Highest score cutoff that still achieves at least ``target_recall``."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    meeting = np.where(tpr >= target_recall)[0]
    if len(meeting) == 0:
        return float(np.min(scores))
    return float(thresholds[meeting[0]])


def threshold_for_fpr(y_true, scores, max_fpr: float = 0.02) -> float:
    """Lowest score cutoff (highest recall) that keeps FPR <= ``max_fpr``."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    allowed = fpr <= max_fpr
    if not allowed.any():
        return float(np.max(scores))
    idx = np.where(allowed)[0]
    best = idx[np.argmax(tpr[idx])]
    return float(thresholds[best])


def _rates(y, scores, thr) -> tuple[float, float]:
    y = np.asarray(y)
    pred = scores >= thr
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    recall = int(((pred == 1) & (y == 1)).sum()) / pos if pos else 0.0
    fpr = int(((pred == 1) & (y == 0)).sum()) / neg if neg else 0.0
    return recall, fpr


def tune_thresholds(
    y_true, scores, target_recall: float = 0.90, max_fpr: float = 0.02
) -> dict[str, float]:
    """Tune both thresholds and report the recall / FPR each one yields."""
    decline = threshold_for_fpr(y_true, scores, max_fpr)
    review = threshold_for_recall(y_true, scores, target_recall)
    # Guard the ordering: review (more lenient) must sit below decline.
    review = min(review, decline)
    r_dec, f_dec = _rates(y_true, scores, decline)
    r_rev, f_rev = _rates(y_true, scores, review)
    return {
        "decline_threshold": decline,
        "review_threshold": review,
        "recall_at_decline": r_dec,
        "fpr_at_decline": f_dec,
        "recall_with_review": r_rev,  # decline+review combined recall
        "fpr_with_review": f_rev,
        "target_recall": target_recall,
        "max_fpr": max_fpr,
    }


@dataclass
class DecisionPolicy:
    """Maps a fraud score to approve / review / decline using two thresholds."""

    decline_threshold: float
    review_threshold: float

    def __post_init__(self):
        # Keep the band coherent even if callers pass them swapped.
        if self.review_threshold > self.decline_threshold:
            self.review_threshold = self.decline_threshold

    def decide_one(self, score: float) -> str:
        if score >= self.decline_threshold:
            return "decline"
        if score >= self.review_threshold:
            return "review"
        return "approve"

    def decide(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        out = np.full(scores.shape, "approve", dtype=object)
        out[scores >= self.review_threshold] = "review"
        out[scores >= self.decline_threshold] = "decline"
        return out

    def to_dict(self) -> dict[str, float]:
        return {
            "decline_threshold": self.decline_threshold,
            "review_threshold": self.review_threshold,
        }
