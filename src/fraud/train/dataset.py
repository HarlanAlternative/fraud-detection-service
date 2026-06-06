"""Shared dataset assembly + time-aware splitting for all training entry points.

Fraud splits must respect time: training on future transactions to predict the past
leaks information and inflates metrics. We sort by ``TransactionDT`` and cut chronological
train / holdout / test slices. ``holdout`` is reserved for threshold tuning so the test
slice stays untouched until final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud.data.load import IEEE_TIME, load_ieee_cis
from fraud.data.synthetic import make_ieee_like
from fraud.features.engineering import build_training_frame


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def summary(self) -> str:
        def fr(y):
            return f"{len(y):,} rows / {y.mean():.2%} fraud"
        return (
            f"train   : {fr(self.y_train)}\n"
            f"holdout : {fr(self.y_holdout)}\n"
            f"test    : {fr(self.y_test)}"
        )


def load_raw(rows: int | None = None, *, synthetic: bool = False, seed: int = 11) -> pd.DataFrame:
    """Return a raw (IEEE-CIS-shaped) frame, falling back to synthetic when data is absent."""
    if synthetic:
        return make_ieee_like(rows or 60_000, seed=seed)
    try:
        return load_ieee_cis(nrows=rows)
    except FileNotFoundError:
        print("[info] IEEE-CIS not downloaded; using synthetic data. See data/README.md.")
        return make_ieee_like(rows or 60_000, seed=seed)


def time_split(
    df_raw: pd.DataFrame,
    *,
    train_frac: float = 0.6,
    holdout_frac: float = 0.2,
) -> Splits:
    """Chronological train / holdout / test split on the curated feature set."""
    ordered = df_raw.sort_values(IEEE_TIME) if IEEE_TIME in df_raw.columns else df_raw
    X, y = build_training_frame(ordered)
    n = len(X)
    i_tr = int(n * train_frac)
    i_ho = int(n * (train_frac + holdout_frac))
    return Splits(
        X_train=X.iloc[:i_tr], y_train=y.iloc[:i_tr],
        X_holdout=X.iloc[i_tr:i_ho], y_holdout=y.iloc[i_tr:i_ho],
        X_test=X.iloc[i_ho:], y_test=y.iloc[i_ho:],
    )
