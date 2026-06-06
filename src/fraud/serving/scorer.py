"""The single inference entry point shared by the API and the Kafka consumer.

Loads a fitted :class:`StackingFraudModel` (from a local joblib file or the MLflow registry),
builds the SHAP reason explainer once, and turns raw transaction records into the public
result contract: ``fraud_score`` + ``fraud_decision`` + ``top_3_reasons`` + ``model_version``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud.config import get_settings
from fraud.explain.shap_reasons import ReasonExplainer
from fraud.features.engineering import prepare_serving_frame
from fraud.models.stacking import StackingFraudModel


class FraudScorer:
    """Wraps a fraud model + reason explainer for low-latency scoring."""

    def __init__(self, model: StackingFraudModel, model_version: str, top_n: int = 3):
        self.model = model
        self.model_version = model_version
        self.explainer = ReasonExplainer(model.xgb, top_n=top_n)

    # --- constructors --------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path | None = None, top_n: int = 3) -> FraudScorer:
        settings = get_settings()
        path = Path(path or settings.models_dir / "stacking.joblib")
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path}. Train one first: python -m fraud.train.pipeline"
            )
        model = StackingFraudModel.load(path)
        return cls(model, model_version=f"{settings.model_name}:{model.version}", top_n=top_n)

    @classmethod
    def from_registry(cls, name: str | None = None, stage: str | None = None,
                      top_n: int = 3) -> FraudScorer:
        """Load the model logged to the MLflow registry (used in containers/Azure)."""
        import mlflow

        settings = get_settings()
        name = name or settings.model_name
        stage = stage or settings.model_stage
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        # We registered a pyfunc wrapping the joblib; pull the underlying StackingFraudModel.
        local = mlflow.artifacts.download_artifacts(f"models:/{name}/{stage}")
        model_path = next(Path(local).rglob("stacking.joblib"))
        model = StackingFraudModel.load(model_path)
        return cls(model, model_version=f"{name}:{stage}", top_n=top_n)

    # --- scoring -------------------------------------------------------------
    def score_frame(self, X: pd.DataFrame) -> list[dict]:
        scores = self.model.predict_proba(X)
        decisions = self.model.policy.decide(scores)
        reasons = self.explainer.explain(X)
        return [
            {
                "fraud_score": round(float(s), 4),
                "fraud_decision": str(d),
                "top_3_reasons": r,
                "model_version": self.model_version,
            }
            for s, d, r in zip(scores, decisions, reasons, strict=True)
        ]

    def score_one(self, record: dict) -> dict:
        return self.score_frame(prepare_serving_frame(record))[0]

    def score_many(self, records: list[dict]) -> list[dict]:
        return self.score_frame(prepare_serving_frame(records))
