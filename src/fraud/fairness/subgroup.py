"""Sub-group performance and disparity analysis.

Fraud models must not systematically disadvantage protected groups (NZ Privacy Act 2020 /
Human Rights Act; banks care deeply). The public benchmark data carries no demographics, so
we attach **synthetic proxy attributes** (age band, region) and measure whether decline rate,
false-positive rate, and recall differ across groups. The headline is the **disparity ratio**
(worst group / best group) per metric — a ratio near 1.0 is equitable; > 1.2 warrants review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.models.metrics import recall_at_fpr

AGE_BANDS = ["18-25", "26-40", "41-60", "60+"]
REGIONS = ["Auckland", "Wellington", "Canterbury", "Otago", "Waikato"]


def add_synthetic_attributes(n: int, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic protected attributes (age band, region) for a fairness audit.

    Independent of the fraud label by construction, so any measured disparity reflects the
    model's behaviour rather than a planted demographic signal.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age_band": rng.choice(AGE_BANDS, size=n, p=[0.22, 0.34, 0.30, 0.14]),
            "region": rng.choice(REGIONS, size=n, p=[0.40, 0.18, 0.18, 0.12, 0.12]),
        }
    )


def subgroup_report(
    y_true, scores, decisions, attributes: pd.DataFrame, max_fpr: float = 0.02
) -> pd.DataFrame:
    """Per-subgroup metrics for every attribute/value in ``attributes``."""
    y = np.asarray(y_true)
    scores = np.asarray(scores)
    decisions = np.asarray(decisions)
    flagged = np.isin(decisions, ["decline", "review"])

    rows = []
    for attr in attributes.columns:
        for group, mask in attributes.groupby(attr).groups.items():
            idx = attributes.index.get_indexer(mask)
            yg, sg, fg = y[idx], scores[idx], flagged[idx]
            neg = (yg == 0)
            rows.append(
                {
                    "attribute": attr,
                    "group": group,
                    "n": int(len(idx)),
                    "fraud_rate": float(yg.mean()),
                    "flag_rate": float(fg.mean()),
                    "fpr": float(fg[neg].mean()) if neg.any() else float("nan"),
                    "recall_at_fpr": recall_at_fpr(yg, sg, max_fpr) if yg.sum() else float("nan"),
                }
            )
    return pd.DataFrame(rows).set_index(["attribute", "group"])


def disparity(report: pd.DataFrame, metric: str = "flag_rate") -> pd.DataFrame:
    """Worst/best ratio of ``metric`` within each attribute (1.0 = perfectly equitable)."""
    out = []
    for attr, sub in report.groupby(level="attribute"):
        vals = sub[metric].dropna()
        if len(vals) < 2 or vals.min() == 0:
            ratio = float("nan")
        else:
            ratio = float(vals.max() / vals.min())
        out.append({"attribute": attr, "metric": metric, "disparity_ratio": ratio,
                    "worst_group": vals.idxmax()[1], "best_group": vals.idxmin()[1]})
    return pd.DataFrame(out).set_index("attribute")
