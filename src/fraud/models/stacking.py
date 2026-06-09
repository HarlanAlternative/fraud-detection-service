"""Stacked fraud ensemble: XGBoost + Isolation Forest + Autoencoder -> LR meta-learner.

Leakage-safe training: the three base learners are fit on the *train* slice; their scores on
an unseen *holdout* slice train the logistic-regression meta-model and tune the decline
threshold. The *test* slice is never touched until final evaluation. The whole object is
self-contained (each base learner carries its own preprocessing), so serving only needs to
hand it a raw curated DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud.models.autoencoder import AEScorer
from fraud.models.iforest import IForestScorer
from fraud.models.threshold import DecisionPolicy, tune_thresholds
from fraud.models.xgb import train_xgb

BASE_ORDER = ["xgboost", "isolation_forest", "autoencoder"]


class StackingFraudModel:
    """The production fraud model: ensemble + meta-learner + decision policy."""

    def __init__(self, target_recall: float = 0.90, max_fpr: float = 0.02):
        self.target_recall = target_recall
        self.max_fpr = max_fpr

        self.xgb = None
        self.iforest = IForestScorer()
        self.autoencoder = AEScorer()
        self.meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.policy: DecisionPolicy | None = None
        self.threshold_info: dict[str, float] = {}
        self.version: str = "0.1.0"

    # --- base scores ---------------------------------------------------------
    def base_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the three base-learner scores as columns (all in [0, 1])."""
        return pd.DataFrame(
            {
                "xgboost": self.xgb.predict_proba(X)[:, 1],
                "isolation_forest": self.iforest.score(X),
                "autoencoder": self.autoencoder.score(X),
            },
            index=X.index,
        )[BASE_ORDER]

    # --- training ------------------------------------------------------------
    def fit(self, X_train, y_train, X_val, y_val) -> StackingFraudModel:
        # 1. base learners on the train slice
        self.xgb = train_xgb(X_train, y_train)
        self.iforest.fit(X_train)
        self.autoencoder.fit(X_train, y_train)

        # 2. meta-learner on holdout base scores (base learners never saw holdout)
        val_base = self.base_scores(X_val)
        self.meta.fit(val_base.to_numpy(), np.asarray(y_val))

        # 3. tune the decline/review thresholds on holdout meta scores
        val_final = self.meta.predict_proba(val_base.to_numpy())[:, 1]
        self.threshold_info = tune_thresholds(
            y_val, val_final, target_recall=self.target_recall, max_fpr=self.max_fpr
        )
        self.policy = DecisionPolicy(
            decline_threshold=self.threshold_info["decline_threshold"],
            review_threshold=self.threshold_info["review_threshold"],
        )
        return self

    # --- inference -----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Final fraud score in [0, 1]."""
        base = self.base_scores(X).to_numpy()
        return self.meta.predict_proba(base)[:, 1]

    def decide(self, X: pd.DataFrame) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("Model is not fitted (no decision policy).")
        return self.policy.decide(self.predict_proba(X))

    def meta_weights(self) -> dict[str, float]:
        """Meta-learner coefficient per base model (interpretability / model card)."""
        coefs = self.meta.coef_.ravel()
        return {name: float(c) for name, c in zip(BASE_ORDER, coefs, strict=True)}

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> StackingFraudModel:
        return joblib.load(path)
