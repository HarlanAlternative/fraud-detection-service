"""Shared pytest fixtures.

A small stacking model is trained once per session on synthetic data (fast autoencoder), and
the FastAPI test client is wired to it by monkeypatching ``FraudScorer.from_path`` so no model
file or database is required.
"""

from __future__ import annotations

import pytest

from fraud.models.autoencoder import AEScorer
from fraud.models.stacking import StackingFraudModel
from fraud.serving.scorer import FraudScorer
from fraud.train.dataset import load_raw, time_split


@pytest.fixture(scope="session")
def splits():
    return time_split(load_raw(8_000, synthetic=True, seed=3))


@pytest.fixture(scope="session")
def model(splits) -> StackingFraudModel:
    m = StackingFraudModel()
    m.autoencoder = AEScorer(epochs=5)  # fewer epochs to keep the test suite quick
    m.fit(splits.X_train, splits.y_train, splits.X_holdout, splits.y_holdout)
    return m


@pytest.fixture(scope="session")
def scorer(model) -> FraudScorer:
    return FraudScorer(model, model_version="test:0.1.0")


@pytest.fixture
def client(scorer, monkeypatch):
    from fastapi.testclient import TestClient

    from fraud.serving import app as app_module

    monkeypatch.setattr(FraudScorer, "from_path", classmethod(lambda cls, *a, **k: scorer))
    with TestClient(app_module.app) as c:
        yield c
