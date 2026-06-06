"""Central configuration for the fraud detection service.

All settings are read from environment variables (prefix ``FRAUD_``) or an ``.env``
file via pydantic-settings, so the same code runs locally, in docker-compose, and on
Azure Container Apps with only env changes. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three levels up from this file (src/fraud/config.py -> repo/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Instantiate via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="FRAUD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- data ---
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    primary_dataset: str = Field(default="ieee-cis")  # ieee-cis | credit-card

    # --- decision policy ---
    target_recall: float = Field(default=0.90, ge=0.0, le=1.0)
    max_fpr: float = Field(default=0.02, ge=0.0, le=1.0)
    review_band: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment: str = Field(default="fraud-detection")
    model_name: str = Field(default="fraud-stacking-ensemble")
    model_stage: str = Field(default="Production")

    # --- serving / logging ---
    database_url: str = Field(
        default="postgresql+psycopg://fraud:fraud@localhost:5432/fraud"
    )
    api_port: int = Field(default=8000)

    # --- streaming ---
    kafka_bootstrap: str = Field(default="localhost:9092")
    kafka_topic: str = Field(default="transactions")
    kafka_group: str = Field(default="fraud-scorer")

    # --- derived paths ---
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    def ensure_dirs(self) -> None:
        """Create the local data/model directories if they do not yet exist."""
        for path in (self.raw_dir, self.interim_dir, self.processed_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (read once per process)."""
    return Settings()
