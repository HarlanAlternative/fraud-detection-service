"""Synthetic IEEE-CIS-shaped data.

Lets the whole pipeline (EDA, training, tests, the API, the Kafka stream) run **before**
the real Kaggle download and inside CI. The generator emits the raw IEEE-CIS column names
so it flows through :func:`fraud.features.engineering.map_ieee_to_curated` unchanged.

The fraud signal is deliberately learnable but noisy: fraud is more likely for large
amounts, certain product codes, the "anonymous"/proton e-mail families, odd hours, and
high transaction-count history. ``drift=True`` shifts the feature distribution so the PSI
monitor and drift alerts have something to fire on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PRODUCTS = ["W", "C", "R", "H", "S"]
_NETWORKS = ["visa", "mastercard", "american express", "discover"]
_CARD_TYPES = ["debit", "credit"]
_EMAILS = ["gmail", "yahoo", "hotmail", "anonymous", "proton", "outlook", "aol"]
_DEVICES = ["desktop", "mobile"]


def make_ieee_like(
    n: int = 20_000,
    *,
    fraud_rate: float = 0.035,
    seed: int = 7,
    drift: bool = False,
) -> pd.DataFrame:
    """Return a synthetic raw IEEE-CIS-style transaction frame with ``n`` rows."""
    rng = np.random.default_rng(seed)

    # Latent fraud propensity built from a few interpretable drivers.
    amt = rng.lognormal(mean=4.0 if not drift else 4.6, sigma=1.1, size=n)
    hour = rng.integers(0, 24, size=n)
    product = rng.choice(_PRODUCTS, size=n, p=[0.55, 0.2, 0.1, 0.1, 0.05])
    email = rng.choice(
        _EMAILS,
        size=n,
        p=[0.42, 0.15, 0.12, 0.08, 0.06, 0.12, 0.05]
        if not drift
        else [0.30, 0.12, 0.10, 0.18, 0.14, 0.10, 0.06],  # more anonymous/proton under drift
    )
    c_hist = rng.poisson(lam=2.0, size=n).astype(float)
    # Pre-generate the remaining numeric arrays so the anomaly component can reference them.
    card1 = rng.integers(1000, 18000, size=n)
    c13 = rng.poisson(lam=3.0, size=n).astype(float)
    d15 = rng.integers(0, 700, size=n).astype(float)
    dist1 = rng.exponential(scale=30, size=n).round(1)

    # --- Component A: a learnable, NON-LINEAR pattern -----------------------------
    # Built from interactions, so gradient-boosted trees capture it but a linear model
    # cannot — this is what makes XGBoost beat logistic regression (as on real fraud).
    big_amt = amt > np.quantile(amt, 0.90)
    anon = np.isin(email, ["anonymous", "proton"])
    night = np.isin(hour, [0, 1, 2, 3, 4])
    prod_c = product == "C"
    hi_hist = c_hist > 4
    z = (
        -0.2
        + 0.6 * big_amt + 0.5 * anon + 0.4 * night + 0.3 * prod_c   # weak main effects
        + 3.0 * (big_amt & anon)                                    # interactions dominate
        + 2.6 * (anon & night)
        + 2.2 * (big_amt & prod_c)
        + 1.8 * (night & hi_hist)
        + rng.normal(0, 0.3, size=n)
    )
    # ~80% of the fraud budget comes from the learnable pattern. Solve the intercept by
    # bisection so the pattern fraud rate hits its target exactly.
    rate_a = fraud_rate * 0.8
    lo, hi = -25.0, 25.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + z)))).mean() > rate_a:
            hi = mid
        else:
            lo = mid
    p_a = 1 / (1 + np.exp(-((lo + hi) / 2 + z)))
    fraud_a = rng.random(n) < p_a

    # --- Component B: rare multivariate anomalies ("zero-day" fraud) ---------------
    # Scattered outliers across several numeric features with no consistent signature,
    # so the supervised model under-fits them while the unsupervised members (Isolation
    # Forest / Autoencoder) flag them as out-of-distribution. This is what lets the
    # stacked ensemble edge past any single model.
    def _z(a):
        a = a.astype(float)
        return (a - a.mean()) / (a.std() + 1e-9)
    extremeness = (
        _z(np.log1p(amt)) ** 2 + _z(c13) ** 2 + _z(d15) ** 2 + _z(card1.astype(float)) ** 2
    )
    cutoff = np.quantile(extremeness, 1 - fraud_rate * 0.2)  # ~20% of the budget
    fraud_b = extremeness >= cutoff

    is_fraud = (fraud_a | fraud_b).astype(int)

    start = rng.integers(86_400, 86_400 * 180, size=n)  # seconds offset, ~6 months span
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "isFraud": is_fraud,
            "TransactionDT": np.sort(start),
            "TransactionAmt": np.round(amt, 2),
            "ProductCD": product,
            "card1": card1,
            "card2": rng.integers(100, 600, size=n).astype(float),
            "card4": rng.choice(_NETWORKS, size=n, p=[0.55, 0.35, 0.06, 0.04]),
            "card6": rng.choice(_CARD_TYPES, size=n, p=[0.6, 0.4]),
            "addr1": rng.integers(100, 540, size=n).astype(float),
            "dist1": dist1,
            "P_emaildomain": [f"{e}.com" for e in email],
            "R_emaildomain": rng.choice([f"{e}.com" for e in _EMAILS] + [None], size=n),
            "C1": c_hist,
            "C2": rng.poisson(lam=1.5, size=n).astype(float),
            "C13": c13,
            "C14": rng.poisson(lam=2.5, size=n).astype(float),
            "D1": rng.integers(0, 600, size=n).astype(float),
            "D4": rng.integers(0, 400, size=n).astype(float),
            "D15": d15,
            "DeviceType": rng.choice(_DEVICES + [None], size=n, p=[0.45, 0.4, 0.15]),
        }
    )
    # Inject realistic missingness into the identity-ish columns.
    for col, frac in (("dist1", 0.55), ("card2", 0.02), ("D4", 0.28)):
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":  # quick smoke check
    d = make_ieee_like(5_000)
    print(d.shape, "fraud rate:", f"{d['isFraud'].mean():.3%}")
    print(d.head(3).to_string())
