"""Full stacked-ensemble training pipeline, tracked end-to-end in MLflow.

    python -m fraud.train.pipeline --rows 200000 --register

Trains the three base learners + LR meta-model, tunes the decision thresholds, evaluates
every model on the untouched test slice, logs params/metrics/thresholds/meta-weights, saves
the model, and logs it as an MLflow pyfunc (registering it when a registry is available).
"""

from __future__ import annotations

import argparse

import mlflow

from fraud.config import get_settings
from fraud.models.metrics import evaluate
from fraud.models.stacking import BASE_ORDER, StackingFraudModel
from fraud.serving.model_wrapper import log_fraud_model
from fraud.train.dataset import load_raw, time_split
from fraud.train.tracking import setup_mlflow


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the stacked fraud ensemble -> MLflow.")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--register", action="store_true", help="register in the MLflow registry")
    parser.add_argument("--target-recall", type=float, default=settings.target_recall)
    parser.add_argument("--max-fpr", type=float, default=settings.max_fpr)
    args = parser.parse_args(argv)

    df = load_raw(args.rows, synthetic=args.synthetic)
    splits = time_split(df)
    print(splits.summary(), "\n")

    setup_mlflow()
    with mlflow.start_run(run_name="stacking-ensemble"):
        model = StackingFraudModel(target_recall=args.target_recall, max_fpr=args.max_fpr)
        model.fit(splits.X_train, splits.y_train, splits.X_holdout, splits.y_holdout)

        # Evaluate the ensemble and each base learner on the held-out test slice.
        test_base = model.base_scores(splits.X_test)
        final = model.predict_proba(splits.X_test)
        final_metrics = evaluate(splits.y_test, final, args.max_fpr)
        metrics = {f"stacking_{k}": v for k, v in final_metrics.items()}
        for name in BASE_ORDER:
            for k, v in evaluate(splits.y_test, test_base[name].to_numpy(), args.max_fpr).items():
                metrics[f"{name}_{k}"] = v

        mlflow.log_params(
            {
                "model": "stacking(xgb+iforest+ae)+lr_meta",
                "n_train": len(splits.y_train),
                "fraud_rate_train": round(float(splits.y_train.mean()), 5),
                "target_recall": args.target_recall,
                "max_fpr": args.max_fpr,
                "data": "synthetic" if args.synthetic else settings.primary_dataset,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"threshold_{k}": v for k, v in model.threshold_info.items()})
        mlflow.log_metrics({f"meta_weight_{k}": v for k, v in model.meta_weights().items()})

        local_path = model.save(settings.models_dir / "stacking.joblib")

        # Save a training reference (features + scores) for the drift monitor.
        reference = splits.X_test.copy()
        reference["fraud_score"] = final
        reference.to_parquet(settings.models_dir / "reference.parquet")

        log_fraud_model(
            str(local_path),
            registered_name=settings.model_name if args.register else None,
        )

        print("test metrics:")
        for k in sorted(metrics):
            print(f"  {k:38s} {metrics[k]:.4f}")
        print("\nthresholds:", {k: round(v, 4) for k, v in model.threshold_info.items()})
        print("meta weights:", {k: round(v, 3) for k, v in model.meta_weights().items()})
        print(f"\nsaved model -> {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
