"""Shared sklearn preprocessing for every base learner.

One fitted ``ColumnTransformer`` is reused across train and serve so the two can never
drift apart. Missing categoricals become an explicit ``"missing"`` category — in fraud,
*absence of a value is itself signal* (e.g. a blank recipient e-mail).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud.features.engineering import feature_lists


def _to_str_missing(df: pd.DataFrame) -> np.ndarray:
    """Cast categorical columns to plain strings, mapping any NA to 'missing'."""
    return (
        df.astype("object")
        .where(df.notna(), other="missing")
        .astype(str)
        .to_numpy()
    )


def build_preprocessor(*, scale: bool = True) -> ColumnTransformer:
    """Return an unfitted preprocessor for the curated feature set.

    Args:
        scale: standard-scale numerics. Needed for the autoencoder / isolation forest /
            logistic regression; harmless for XGBoost. Set ``False`` for tree-only use.
    """
    cols = feature_lists()

    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)

    categorical = Pipeline(
        [
            ("to_str", FunctionTransformer(_to_str_missing, feature_names_out="one-to-one")),
            # min_frequency caps cardinality (e.g. card networks) and folds rares into one col.
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, cols["numeric"]),
            ("cat", categorical, cols["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
