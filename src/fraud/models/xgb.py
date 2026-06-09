"""XGBoost base learner — the supervised workhorse and the industry-standard baseline."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud.models.preprocess import build_preprocessor


def build_xgb(scale_pos_weight: float = 1.0, **overrides) -> Pipeline:
    """Return an unfitted ``preprocess -> XGBClassifier`` pipeline."""
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=2,
        scale_pos_weight=scale_pos_weight,  # counteract class imbalance without resampling
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    params.update(overrides)
    # Trees don't need scaled inputs; skip it for speed and to keep raw feature semantics.
    return Pipeline([("pre", build_preprocessor(scale=False)), ("clf", XGBClassifier(**params))])


def train_xgb(X: pd.DataFrame, y: pd.Series, **overrides) -> Pipeline:
    """Fit XGBoost with class-imbalance weighting derived from ``y``."""
    pos = max(int(y.sum()), 1)
    neg = int((y == 0).sum())
    model = build_xgb(scale_pos_weight=neg / pos, **overrides)
    model.fit(X, y)
    return model
