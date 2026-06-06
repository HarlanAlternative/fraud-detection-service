import numpy as np

from fraud.data.schema import CURATED_FEATURES
from fraud.data.synthetic import make_ieee_like
from fraud.features.engineering import build_training_frame, prepare_serving_frame


def test_build_training_frame_shapes():
    df = make_ieee_like(2_000, seed=1)
    X, y = build_training_frame(df)
    assert list(X.columns) == CURATED_FEATURES
    assert len(X) == len(y) == 2_000
    assert set(np.unique(y)).issubset({0, 1})


def test_email_normalisation():
    X, _ = build_training_frame(make_ieee_like(500, seed=2))
    # families, not full domains (regex=False: '.' is a literal dot, not a wildcard)
    assert X["purchaser_email_domain"].dropna().str.contains(".", regex=False).sum() == 0


def test_serving_frame_derives_time_and_fills_missing():
    rec = {"transaction_amt": 50.0, "transaction_dt": "2024-05-01T03:30:00Z"}
    sf = prepare_serving_frame(rec)
    assert list(sf.columns) == CURATED_FEATURES
    assert sf["hour"].iloc[0] == 3
    assert sf["day_of_week"].iloc[0] == 2  # 2024-05-01 is a Wednesday


def test_serving_frame_accepts_list():
    sf = prepare_serving_frame([{"transaction_amt": 1.0}, {"transaction_amt": 2.0}])
    assert len(sf) == 2
