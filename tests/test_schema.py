import pandas as pd
import pandera.errors as pa_errors
import pytest

from fraud.data.schema import (
    CURATED_FEATURES,
    credit_card_schema,
    ieee_transaction_schema,
    scoring_input_schema,
)


def test_curated_feature_count():
    assert len(CURATED_FEATURES) == 20
    assert "transaction_amt" in CURATED_FEATURES


def test_scoring_schema_accepts_valid_row():
    df = pd.DataFrame([{"transaction_amt": 100.0, "product_cd": "C"}])
    validated = scoring_input_schema().validate(df)
    assert validated["transaction_amt"].iloc[0] == 100.0


def test_scoring_schema_rejects_negative_amount():
    df = pd.DataFrame([{"transaction_amt": -5.0}])
    with pytest.raises(pa_errors.SchemaError):
        scoring_input_schema().validate(df)


def test_ieee_schema_rejects_non_binary_target():
    df = pd.DataFrame(
        {"TransactionID": [1, 2], "isFraud": [0, 5], "TransactionDT": [1, 2],
         "TransactionAmt": [1.0, 2.0]}
    )
    with pytest.raises(pa_errors.SchemaError):
        ieee_transaction_schema().validate(df)


def test_credit_card_schema_ok():
    df = pd.DataFrame({"Class": [0, 1], "Amount": [1.0, 2.0], "Time": [0.0, 1.0]})
    credit_card_schema().validate(df)
