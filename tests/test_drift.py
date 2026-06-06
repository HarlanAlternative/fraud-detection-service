import numpy as np

from fraud.data.schema import CURATED_CATEGORICAL, CURATED_FEATURES
from fraud.data.synthetic import make_ieee_like
from fraud.features.engineering import build_training_frame
from fraud.monitoring.drift import classify_psi, feature_psi, psi, rolling_recall


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=10_000)
    b = rng.normal(size=10_000)
    assert psi(a, b) < 0.1


def test_psi_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=10_000)
    b = rng.normal(2, 1, size=10_000)  # mean-shifted
    assert psi(a, b) > 0.2


def test_feature_psi_flags_drifted_synthetic():
    ref = build_training_frame(make_ieee_like(8_000, seed=1))[0]
    cur = build_training_frame(make_ieee_like(8_000, seed=9, drift=True))[0]
    psis = feature_psi(ref, cur, CURATED_FEATURES, set(CURATED_CATEGORICAL))
    assert max(psis.values()) > 0.2  # drift is detectable


def test_classify_psi_buckets():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.3) == "significant"


def test_rolling_recall():
    rows = [
        {"label": 1, "fraud_decision": "decline"},
        {"label": 1, "fraud_decision": "approve"},
        {"label": 0, "fraud_decision": "approve"},
    ]
    assert rolling_recall(rows) == 0.5
