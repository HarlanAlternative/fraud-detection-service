"""SHAP-based reason codes.

Every decline/review needs a human-readable justification ("why was this flagged?"). We run
a fast ``TreeExplainer`` on the **XGBoost base learner** (per the brief: explain the base
learners, not the meta-model), aggregate the one-hot SHAP contributions back to the original
curated feature, and surface the top-N drivers. Aggregating to curated features is what makes
the output readable ("purchaser_email_domain" rather than three "cat__..._x" columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from fraud.data.schema import CURATED_CATEGORICAL
from fraud.features.engineering import feature_lists


def _source_feature(encoded_name: str) -> str:
    """Map an encoded column name (e.g. 'cat__product_cd_w') to its curated source feature."""
    name = encoded_name.split("__", 1)[-1]  # drop the 'num__' / 'cat__' transformer prefix
    for cat in CURATED_CATEGORICAL:
        if name == cat or name.startswith(cat + "_"):
            return cat
    return name  # numeric features map to themselves


class ReasonExplainer:
    """Produces top-N reason codes from a fitted XGBoost pipeline."""

    def __init__(self, xgb_pipeline, top_n: int = 3):
        self.pre = xgb_pipeline.named_steps["pre"]
        self.clf = xgb_pipeline.named_steps["clf"]
        self.top_n = top_n
        self.explainer = shap.TreeExplainer(self.clf)

        encoded = list(self.pre.get_feature_names_out())
        self._encoded = encoded
        self._sources = [_source_feature(n) for n in encoded]
        self._all_features = feature_lists()["all"]

    def _shap_matrix(self, Z) -> np.ndarray:
        vals = self.explainer.shap_values(Z)
        if isinstance(vals, list):          # older API: [class0, class1]
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:                  # (n, features, classes)
            vals = vals[..., -1]
        return vals

    def explain(self, X: pd.DataFrame) -> list[list[dict]]:
        """Return, per row, a list of top-N reason dicts sorted by absolute SHAP impact."""
        Z = self.pre.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        shap_vals = self._shap_matrix(Z)

        # Aggregate signed SHAP per curated source feature.
        sources = np.array(self._sources)
        unique = list(dict.fromkeys(self._sources))
        results: list[list[dict]] = []
        for i in range(shap_vals.shape[0]):
            agg = {src: float(shap_vals[i, sources == src].sum()) for src in unique}
            ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[: self.top_n]
            row_reasons = []
            for feat, impact in ranked:
                value = X.iloc[i][feat] if feat in X.columns else None
                row_reasons.append(
                    {
                        "feature": feat,
                        "value": None if pd.isna(value) else _clean(value),
                        "impact": round(impact, 4),
                        "direction": "increases_risk" if impact > 0 else "decreases_risk",
                    }
                )
            results.append(row_reasons)
        return results


def _clean(value):
    """JSON-friendly scalar."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    return str(value)
