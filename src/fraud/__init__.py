"""Fraud / anomaly detection service.

A stacked ensemble (XGBoost + Isolation Forest + Autoencoder, LR meta-learner) for
real-time credit-card fraud scoring, served behind a FastAPI API with SHAP reason
codes, MLflow model registry, and PSI-based drift monitoring.
"""

__version__ = "0.1.0"
