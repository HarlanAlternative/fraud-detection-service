"""MLflow pyfunc wrapper around :class:`StackingFraudModel`.

Logging the model through pyfunc gives us a versioned, registry-promotable artifact whose
``predict`` accepts raw curated transaction records and returns score + decision. The
FastAPI service loads the same wrapper so training and serving share one inference path.
"""

from __future__ import annotations

import mlflow
import pandas as pd


class FraudModel(mlflow.pyfunc.PythonModel):
    """pyfunc flavour: raw transaction records -> ``{fraud_score, fraud_decision}``."""

    def load_context(self, context):
        from fraud.models.stacking import StackingFraudModel

        self.model = StackingFraudModel.load(context.artifacts["model_path"])

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        from fraud.features.engineering import prepare_serving_frame

        X = prepare_serving_frame(model_input)
        score = self.model.predict_proba(X)
        decision = self.model.policy.decide(score)
        return pd.DataFrame({"fraud_score": score, "fraud_decision": decision})


def log_fraud_model(
    model_path: str, artifact_path: str = "model", registered_name: str | None = None
):
    """Log a saved StackingFraudModel as a pyfunc, optionally registering it.

    Registration requires a database-backed MLflow store; with the local file store it is
    skipped with a warning so training never fails offline.
    """
    kwargs = dict(
        name=artifact_path,
        python_model=FraudModel(),
        artifacts={"model_path": model_path},
        pip_requirements=["scikit-learn", "xgboost", "torch", "pandas", "joblib"],
    )
    if registered_name:
        try:
            return mlflow.pyfunc.log_model(registered_model_name=registered_name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] registry unavailable ({exc}); logging without registration.")
    return mlflow.pyfunc.log_model(**kwargs)
