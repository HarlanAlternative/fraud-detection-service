"""MLflow setup shared by training scripts.

Uses the configured tracking server when reachable, otherwise transparently falls back to
a local ``./mlruns`` file store so training always works offline (and in CI).
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import mlflow

from fraud.config import get_settings


def _reachable(uri: str, timeout: float = 1.5) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        return True  # file:/sqlite stores are always "reachable"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_mlflow(experiment: str | None = None) -> str:
    """Configure MLflow tracking + experiment. Returns the tracking URI actually used."""
    settings = get_settings()
    uri = settings.mlflow_tracking_uri
    if not _reachable(uri):
        uri = "file:./mlruns"
        print(f"[info] MLflow server unreachable; logging to local store '{uri}'.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or settings.mlflow_experiment)
    return uri
