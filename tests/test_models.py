import numpy as np

from fraud.models.stacking import BASE_ORDER, StackingFraudModel
from fraud.models.threshold import DecisionPolicy, tune_thresholds


def test_predict_proba_in_unit_interval(model, splits):
    p = model.predict_proba(splits.X_test)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert len(p) == len(splits.X_test)


def test_decisions_are_valid_labels(model, splits):
    decisions = set(model.decide(splits.X_test).tolist())
    assert decisions.issubset({"approve", "review", "decline"})


def test_base_scores_columns(model, splits):
    base = model.base_scores(splits.X_test.head(20))
    assert list(base.columns) == BASE_ORDER
    assert ((base >= 0) & (base <= 1)).all().all()


def test_stacking_is_competitive(model, splits):
    from sklearn.metrics import roc_auc_score

    base = model.base_scores(splits.X_test)
    final = model.predict_proba(splits.X_test)
    auc_stack = roc_auc_score(splits.y_test, final)
    auc_xgb = roc_auc_score(splits.y_test, base["xgboost"])
    # Smoke check only: the ensemble clearly learns and stays close to its strongest base
    # learner. (The genuine ensemble win is small and data-size-dependent; it's demonstrated
    # on the full dataset in notebooks/03_benchmark, not asserted on tiny test data.)
    assert auc_stack > 0.70
    assert auc_stack >= auc_xgb - 0.06


def test_save_load_roundtrip(model, splits, tmp_path):
    p = model.save(tmp_path / "m.joblib")
    reloaded = StackingFraudModel.load(p)
    a = model.predict_proba(splits.X_test.head(30))
    b = reloaded.predict_proba(splits.X_test.head(30))
    assert np.allclose(a, b)


def test_decision_policy_ordering():
    policy = DecisionPolicy(decline_threshold=0.8, review_threshold=0.3)
    assert policy.decide_one(0.9) == "decline"
    assert policy.decide_one(0.5) == "review"
    assert policy.decide_one(0.1) == "approve"


def test_tune_thresholds_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    scores = np.clip(0.15 * y + rng.random(5000) * 0.3, 0, 1)
    info = tune_thresholds(y, scores, target_recall=0.9, max_fpr=0.02)
    assert info["fpr_at_decline"] <= 0.05  # near the 2% budget (sampling slack)
    assert info["review_threshold"] <= info["decline_threshold"]
