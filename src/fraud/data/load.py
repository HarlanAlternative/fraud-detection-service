"""Load the raw fraud datasets into pandas DataFrames.

* IEEE-CIS (primary): joins ``train_transaction`` with ``train_identity`` on
  ``TransactionID`` and derives a real timestamp from ``TransactionDT``.
* Credit Card Fraud: the single ``creditcard.csv`` (baseline + unit-test fixture).

Run ``python -m fraud.data.load --check`` to verify which datasets are present.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from fraud.config import get_settings

# --- IEEE-CIS conventions -------------------------------------------------------
IEEE_TARGET = "isFraud"
IEEE_ID = "TransactionID"
IEEE_TIME = "TransactionDT"  # seconds offset from a fixed reference instant
IEEE_AMOUNT = "TransactionAmt"
# The competition never published the absolute start time; 2017-12-01 is the widely
# used community reference that makes the day-of-week / hour patterns line up.
IEEE_REFERENCE_DATE = pd.Timestamp("2017-12-01")

# --- Credit Card Fraud conventions ----------------------------------------------
CC_TARGET = "Class"
CC_AMOUNT = "Amount"
CC_TIME = "Time"  # seconds elapsed since the first transaction in the dataset


def _require(path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset file not found: {path}\n"
            "Download it first: see data/README.md and `python -m fraud.data.download`."
        )


def add_transaction_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``transaction_dt`` timestamp column derived from ``TransactionDT`` seconds."""
    out = df.copy()
    out["transaction_dt"] = IEEE_REFERENCE_DATE + pd.to_timedelta(out[IEEE_TIME], unit="s")
    out["hour"] = out["transaction_dt"].dt.hour
    out["day_of_week"] = out["transaction_dt"].dt.dayofweek
    return out


def load_ieee_cis(
    *,
    nrows: int | None = None,
    usecols: list[str] | None = None,
    with_identity: bool = True,
    add_datetime: bool = True,
) -> pd.DataFrame:
    """Load and join the IEEE-CIS training tables.

    Args:
        nrows: optionally cap the number of transaction rows (handy for quick EDA).
        usecols: optional subset of transaction columns to read (TransactionID and the
            target are always kept).
        with_identity: left-join ``train_identity`` (device / id_* features).
        add_datetime: derive ``transaction_dt`` / ``hour`` / ``day_of_week``.
    """
    settings = get_settings()
    root = settings.raw_dir / "ieee-cis"
    tx_path = root / "train_transaction.csv"
    _require(tx_path)

    if usecols is not None:
        usecols = sorted({IEEE_ID, IEEE_TARGET, IEEE_TIME, *usecols})

    tx = pd.read_csv(tx_path, nrows=nrows, usecols=usecols)

    if with_identity:
        id_path = root / "train_identity.csv"
        if id_path.exists():
            identity = pd.read_csv(id_path)
            tx = tx.merge(identity, on=IEEE_ID, how="left")
        else:
            print(f"[warn] {id_path.name} missing; continuing with transaction features only.")

    if add_datetime:
        tx = add_transaction_datetime(tx)
    return tx


def load_credit_card(*, nrows: int | None = None) -> pd.DataFrame:
    """Load the Kaggle Credit Card Fraud dataset (``creditcard.csv``)."""
    settings = get_settings()
    path = settings.raw_dir / "credit-card" / "creditcard.csv"
    _require(path)
    return pd.read_csv(path, nrows=nrows)


def load_primary(**kwargs) -> tuple[pd.DataFrame, str]:
    """Load whichever dataset ``settings.primary_dataset`` points at.

    Returns ``(dataframe, target_column)`` so callers stay dataset-agnostic.
    """
    settings = get_settings()
    if settings.primary_dataset == "ieee-cis":
        return load_ieee_cis(**kwargs), IEEE_TARGET
    if settings.primary_dataset == "credit-card":
        return load_credit_card(**kwargs), CC_TARGET
    raise ValueError(f"Unknown primary_dataset: {settings.primary_dataset!r}")


def _check() -> int:
    """Report presence + basic stats for each dataset without loading everything."""
    settings = get_settings()
    print(f"data dir: {settings.data_dir}\n")
    ok = True
    for key, sentinel in (
        ("ieee-cis", "train_transaction.csv"),
        ("credit-card", "creditcard.csv"),
        ("paysim", "PS_20174392719_1491204439457_log.csv"),
    ):
        path = settings.raw_dir / key / sentinel
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"[present] {key:<12} {sentinel} ({size_mb:,.0f} MB)")
        else:
            ok = False
            print(f"[missing] {key:<12} {sentinel}")

    # Quick fraud-rate peek on the primary dataset if present.
    try:
        df, target = load_primary(nrows=50_000, with_identity=False)
        rate = df[target].mean()
        print(f"\nprimary='{settings.primary_dataset}' first-50k fraud rate: {rate:.4%}")
    except FileNotFoundError:
        print("\nprimary dataset not downloaded yet; skipping fraud-rate peek.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect / verify the local fraud datasets.")
    parser.add_argument("--check", action="store_true", help="report which datasets are present")
    args = parser.parse_args(argv)
    if args.check:
        return _check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
