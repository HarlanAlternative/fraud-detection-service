"""Train the XGBoost baseline and log it to MLflow.

    python -m fraud.train.baseline --rows 200000

This is the Phase-1 baseline; the full stacking pipeline lives in ``fraud.train.pipeline``.
"""

from __future__ import annotations

import argparse

import mlflow

from fraud.models.metrics import evaluate
from fraud.models.xgb import train_xgb
from fraud.train.dataset import load_raw, time_split
from fraud.train.tracking import setup_mlflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="XGBoost fraud baseline -> MLflow.")
    parser.add_argument("--rows", type=int, default=None, help="cap rows (None = all)")
    parser.add_argument("--synthetic", action="store_true", help="force synthetic data")
    parser.add_argument("--max-fpr", type=float, default=0.02)
    args = parser.parse_args(argv)

    df = load_raw(args.rows, synthetic=args.synthetic)
    splits = time_split(df)
    print(splits.summary())

    setup_mlflow()
    with mlflow.start_run(run_name="xgb-baseline"):
        model = train_xgb(splits.X_train, splits.y_train)

        test_scores = model.predict_proba(splits.X_test)[:, 1]
        metrics = evaluate(splits.y_test, test_scores, max_fpr=args.max_fpr)

        mlflow.log_params(
            {
                "model": "xgboost",
                "n_train": len(splits.y_train),
                "fraud_rate_train": round(float(splits.y_train.mean()), 5),
                "max_fpr": args.max_fpr,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        print("\ntest metrics:")
        for k, v in metrics.items():
            print(f"  {k:28s} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
