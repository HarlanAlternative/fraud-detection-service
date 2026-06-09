"""Isolation Forest base learner — unsupervised, catches zero-day fraud patterns.

Trained without labels, so it can flag novel fraud the supervised XGBoost never saw in
training. ``score`` returns a calibrated 0-1 anomaly score (1 = most anomalous), with the
calibration range fixed from the training fold so train/serve stay consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from fraud.models.preprocess import build_preprocessor


class IForestScorer:
    """Unsupervised anomaly scorer with a stable 0-1 output."""

    def __init__(self, n_estimators: int = 200, contamination: float | str = "auto"):
        self.pipeline = Pipeline(
            [
                ("pre", build_preprocessor(scale=True)),
                (
                    "iforest",
                    IsolationForest(
                        n_estimators=n_estimators,
                        contamination=contamination,
                        max_samples="auto",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self._lo: float = 0.0
        self._hi: float = 1.0

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        # decision_function: higher = more normal. Negate so higher = more anomalous.
        return -self.pipeline.named_steps["iforest"].decision_function(
            self.pipeline.named_steps["pre"].transform(X)
        )

    def fit(self, X: pd.DataFrame, y=None) -> IForestScorer:
        self.pipeline.fit(X)  # y ignored (unsupervised)
        raw = self._raw(X)
        # Calibrate to the 1st/99th pct of training scores (robust min-max).
        self._lo, self._hi = np.percentile(raw, [1, 99])
        if self._hi <= self._lo:
            self._hi = self._lo + 1e-9
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1] (1 = most anomalous)."""
        raw = self._raw(X)
        return np.clip((raw - self._lo) / (self._hi - self._lo), 0.0, 1.0)
