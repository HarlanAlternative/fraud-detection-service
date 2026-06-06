"""Pandera schemas + the canonical curated-feature contract.

Design decision: rather than feed all ~430 anonymised IEEE-CIS columns to the model, we
score on a **curated, interpretable feature set**. This keeps SHAP reason codes
human-readable ("purchaser_email_domain", "transaction_amt") instead of opaque ("V258"),
which is the whole point of regulator-friendly explainability. These constants are the
single source of truth shared by feature engineering, the model, and the serving API.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

# --- curated feature contract ---------------------------------------------------
# Interpretable numeric features. C*/D* are IEEE-CIS user-history counts/timedeltas.
CURATED_NUMERIC: list[str] = [
    "transaction_amt",   # TransactionAmt
    "dist1",             # distance between billing / shipping-ish signal
    "card1",             # card identifier (high-signal numeric)
    "card2",
    "addr1",             # billing region code
    "C1", "C2", "C13", "C14",   # transaction-count aggregates (user history)
    "D1", "D4", "D15",          # days since prior events (user history)
]

# Interpretable categoricals.
CURATED_CATEGORICAL: list[str] = [
    "product_cd",               # ProductCD
    "card_network",             # card4: visa / mastercard / amex / discover
    "card_type",                # card6: debit / credit
    "purchaser_email_domain",   # P_emaildomain
    "recipient_email_domain",   # R_emaildomain
    "device_type",              # DeviceType
]

# Derived server-side from the timestamp (not supplied by the caller).
DERIVED_NUMERIC: list[str] = ["hour", "day_of_week"]

CURATED_FEATURES: list[str] = CURATED_NUMERIC + CURATED_CATEGORICAL + DERIVED_NUMERIC

# Maps a curated name -> its raw IEEE-CIS column (used by feature engineering).
IEEE_COLUMN_MAP: dict[str, str] = {
    "transaction_amt": "TransactionAmt",
    "dist1": "dist1",
    "card1": "card1",
    "card2": "card2",
    "addr1": "addr1",
    "C1": "C1", "C2": "C2", "C13": "C13", "C14": "C14",
    "D1": "D1", "D4": "D4", "D15": "D15",
    "product_cd": "ProductCD",
    "card_network": "card4",
    "card_type": "card6",
    "purchaser_email_domain": "P_emaildomain",
    "recipient_email_domain": "R_emaildomain",
    "device_type": "DeviceType",
}


# --- training-data sanity schemas (lenient; only the columns we depend on) -------
def ieee_transaction_schema() -> DataFrameSchema:
    """Sanity checks on raw IEEE-CIS after join: target binary, amount/time non-negative."""
    return DataFrameSchema(
        {
            "TransactionID": Column(int, unique=True),
            "isFraud": Column(int, pa.Check.isin([0, 1])),
            "TransactionDT": Column(int, pa.Check.ge(0)),
            "TransactionAmt": Column(float, pa.Check.ge(0)),
        },
        coerce=True,
        strict=False,  # hundreds of other columns are allowed through untouched
    )


def credit_card_schema() -> DataFrameSchema:
    """Sanity checks on the Kaggle Credit Card Fraud dataset."""
    return DataFrameSchema(
        {
            "Class": Column(int, pa.Check.isin([0, 1])),
            "Amount": Column(float, pa.Check.ge(0)),
            "Time": Column(float, pa.Check.ge(0)),
        },
        coerce=True,
        strict=False,
    )


# --- serving input schema (the public API contract) -----------------------------
def scoring_input_schema() -> DataFrameSchema:
    """Validate an incoming transaction (one row) before scoring.

    Numeric history aggregates and ``dist1`` are nullable (a brand-new card legitimately
    has no history). Categoricals are nullable and normalised downstream. The amount is
    the one hard constraint: it must be present and non-negative.
    """
    nullable_num = {
        name: Column(float, nullable=True, coerce=True, required=False)
        for name in CURATED_NUMERIC
        if name != "transaction_amt"
    }
    nullable_cat = {
        name: Column(str, nullable=True, coerce=True, required=False)
        for name in CURATED_CATEGORICAL
    }
    return DataFrameSchema(
        {
            "transaction_amt": Column(float, pa.Check.ge(0), coerce=True),
            **nullable_num,
            **nullable_cat,
        },
        strict=False,
        coerce=True,
    )
