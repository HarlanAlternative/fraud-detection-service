"""Feature engineering.

Two entry points share one definition of "the features":

* :func:`build_training_frame` — raw IEEE-CIS (joined) -> ``(X, y)`` on the curated set.
* :func:`prepare_serving_frame` — curated-named records (from the API / Kafka) -> ``X``.

Categorical *encoding* and numeric *scaling* are intentionally left to the model's
sklearn pipeline (``fraud.models``) so that train and serve share one fitted transformer
and cannot drift apart. This module only selects, renames, cleans, and adds time features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.data.load import IEEE_TARGET, add_transaction_datetime
from fraud.data.schema import (
    CURATED_CATEGORICAL,
    CURATED_FEATURES,
    CURATED_NUMERIC,
    DERIVED_NUMERIC,
    IEEE_COLUMN_MAP,
)


def _normalise_email(series: pd.Series) -> pd.Series:
    """Lower-case and collapse rare e-mail providers to their family (interpretable)."""
    s = series.astype("string").str.lower()
    # Group the long tail: gmail.com/gmail.fr -> "gmail", etc. Keeps reason codes readable.
    family = s.str.split(".").str[0]
    return family.where(~family.isna(), other=pd.NA)


def map_ieee_to_curated(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw IEEE-CIS columns we use to their curated names + add time features."""
    if "transaction_dt" not in df.columns and "TransactionDT" in df.columns:
        df = add_transaction_datetime(df)

    out = pd.DataFrame(index=df.index)
    for curated, raw in IEEE_COLUMN_MAP.items():
        out[curated] = df[raw] if raw in df.columns else np.nan

    # Interpretable cleanups on the categoricals.
    out["purchaser_email_domain"] = _normalise_email(out["purchaser_email_domain"])
    out["recipient_email_domain"] = _normalise_email(out["recipient_email_domain"])
    for col in ("product_cd", "card_network", "card_type", "device_type"):
        out[col] = out[col].astype("string").str.lower()

    # Derived time features (added by add_transaction_datetime).
    out["hour"] = df.get("hour", pd.Series(np.nan, index=df.index))
    out["day_of_week"] = df.get("day_of_week", pd.Series(np.nan, index=df.index))

    if IEEE_TARGET in df.columns:
        out[IEEE_TARGET] = df[IEEE_TARGET].astype(int)
    return out


def build_training_frame(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn raw (joined) IEEE-CIS into ``(X[CURATED_FEATURES], y)``."""
    curated = map_ieee_to_curated(df_raw)
    if IEEE_TARGET not in curated.columns:
        raise ValueError("Training frame requires the 'isFraud' target column.")
    y = curated[IEEE_TARGET].astype(int)
    x = curated[CURATED_FEATURES].copy()
    return x, y


def prepare_serving_frame(records: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    """Coerce inbound transaction records (curated names) into an ``X[CURATED_FEATURES]`` frame.

    Accepts a single dict, a list of dicts, or a DataFrame. Missing curated columns are
    added as NaN (the model pipeline imputes), and time features are derived from
    ``transaction_dt`` if present.
    """
    if isinstance(records, dict):
        df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        df = records.copy()

    # Derive hour/day_of_week from a supplied timestamp when not already given.
    if "transaction_dt" in df.columns:
        ts = pd.to_datetime(df["transaction_dt"], errors="coerce", utc=True)
        df["hour"] = df.get("hour", ts.dt.hour)
        df["day_of_week"] = df.get("day_of_week", ts.dt.dayofweek)

    # Normalise categoricals the same way as training.
    if "purchaser_email_domain" in df.columns:
        df["purchaser_email_domain"] = _normalise_email(df["purchaser_email_domain"])
    if "recipient_email_domain" in df.columns:
        df["recipient_email_domain"] = _normalise_email(df["recipient_email_domain"])
    for col in ("product_cd", "card_network", "card_type", "device_type"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.lower()

    # Ensure every model feature exists, in the right order.
    for col in CURATED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df[CURATED_FEATURES].copy()


def feature_lists() -> dict[str, list[str]]:
    """Convenience accessor for downstream pipelines (encoders/scalers)."""
    return {
        "numeric": CURATED_NUMERIC + DERIVED_NUMERIC,
        "categorical": CURATED_CATEGORICAL,
        "all": CURATED_FEATURES,
    }
