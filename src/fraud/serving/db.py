"""Inference logging to Postgres, with ground-truth backfill.

Every score is logged so we can (a) monitor the live prediction distribution and (b) compute
**recall on a rolling window once labels arrive** — fraud labels lag by days (chargebacks),
so ``backfill_label`` updates rows after the fact. The logger degrades gracefully to a no-op
when no database is reachable, so the API and tests run without Postgres.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)

from fraud.config import get_settings

_metadata = MetaData()

inference_log = Table(
    "inference_log",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("transaction_id", String, index=True),
    Column("scored_at", DateTime, default=dt.datetime.utcnow),
    Column("fraud_score", Float),
    Column("fraud_decision", String),
    Column("model_version", String),
    Column("features", JSON),
    Column("label", Integer, nullable=True),  # backfilled ground truth (0/1)
)


def _jsonable(features: dict) -> dict:
    """Coerce a (possibly numpy/NaN-laden) feature row to JSON-safe native types."""
    out: dict = {}
    for k, v in features.items():
        if v is None or (np.isscalar(v) and pd.isna(v)):
            out[k] = None
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


class InferenceLogger:
    """Best-effort logger; never raises into the request path."""

    def __init__(self, database_url: str | None = None):
        self.enabled = False
        self.engine = None
        url = database_url or get_settings().database_url
        try:
            self.engine = create_engine(url, pool_pre_ping=True, future=True)
            _metadata.create_all(self.engine)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] inference logging disabled (no DB): {exc}")

    def log(self, features: dict, result: dict) -> None:
        if not self.enabled:
            return
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    inference_log.insert().values(
                        transaction_id=result.get("transaction_id"),
                        fraud_score=result["fraud_score"],
                        fraud_decision=result["fraud_decision"],
                        model_version=result["model_version"],
                        features=_jsonable(features),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] inference log insert failed: {exc}")

    def backfill_label(self, transaction_id: str, label: int) -> int:
        """Set the ground-truth label for a previously scored transaction. Returns rows updated."""
        if not self.enabled:
            return 0
        with self.engine.begin() as conn:
            res = conn.execute(
                inference_log.update()
                .where(inference_log.c.transaction_id == transaction_id)
                .values(label=label)
            )
            return res.rowcount

    def recent_labeled(self, limit: int = 5000):
        """Return recent rows that have a backfilled label (for rolling-recall drift)."""
        if not self.enabled:
            return []
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT fraud_score, fraud_decision, label FROM inference_log "
                    "WHERE label IS NOT NULL ORDER BY scored_at DESC LIMIT :lim"
                ),
                {"lim": limit},
            ).fetchall()
            return [dict(r._mapping) for r in rows]
