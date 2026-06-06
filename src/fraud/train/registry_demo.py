"""Demonstrate an MLflow Model Registry v1 -> v2 promotion.

Trains two versions of the stacking ensemble, registers both, and promotes the better one to
the ``champion`` alias (archiving the old champion) — the exact lifecycle a team runs when
shipping a retrained model. Uses modern registry **aliases** (champion / challenger) rather
than the deprecated stages, and a SQLite-backed tracking store so it runs offline.

    python -m fraud.train.registry_demo
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from fraud.config import get_settings
from fraud.models.metrics import recall_at_fpr
from fraud.models.stacking import StackingFraudModel
from fraud.serving.model_wrapper import log_fraud_model
from fraud.train.dataset import load_raw, time_split


def _train_and_register(name: str, seed: int, rows: int, max_fpr: float) -> tuple[int, float]:
    """Train one model version, log + register it, return ``(version, recall_at_fpr)``."""
    splits = time_split(load_raw(rows, synthetic=True, seed=seed))
    model = StackingFraudModel()
    model.fit(splits.X_train, splits.y_train, splits.X_holdout, splits.y_holdout)
    score = recall_at_fpr(splits.y_test, model.predict_proba(splits.X_test), max_fpr)

    with mlflow.start_run(run_name=f"{name}-seed{seed}"):
        mlflow.log_metric("recall_at_fpr", score)
        mlflow.log_param("seed", seed)
        with tempfile.TemporaryDirectory() as tmp:
            path = model.save(Path(tmp) / "stacking.joblib")
            log_fraud_model(str(path), registered_name=name)
    # The registered version number is the latest for this name.
    client = MlflowClient()
    version = max(int(mv.version) for mv in client.search_model_versions(f"name='{name}'"))
    return version, score


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="MLflow registry v1->v2 promotion demo.")
    parser.add_argument("--rows", type=int, default=40_000)
    parser.add_argument("--name", default=settings.model_name)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow_registry.db")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fraud-registry-demo")
    client = MlflowClient()
    print(f"tracking: {args.tracking_uri}\n")

    # --- v1: first model, becomes champion ---
    v1, r1 = _train_and_register(args.name, seed=1, rows=args.rows, max_fpr=settings.max_fpr)
    client.set_registered_model_alias(args.name, "champion", v1)
    print(f"registered v{v1} (recall@fpr={r1:.4f}) -> alias 'champion'")

    # --- v2: retrained candidate ---
    v2, r2 = _train_and_register(args.name, seed=2, rows=args.rows, max_fpr=settings.max_fpr)
    client.set_registered_model_alias(args.name, "challenger", v2)
    print(f"registered v{v2} (recall@fpr={r2:.4f}) -> alias 'challenger'")

    # --- promotion decision ---
    print("\n--- promotion gate ---")
    if r2 >= r1:
        client.set_registered_model_alias(args.name, "champion", v2)
        client.set_registered_model_tag(args.name, "last_promotion", f"v{v1}->v{v2}")
        client.set_model_version_tag(args.name, str(v1), "status", "archived")
        client.delete_registered_model_alias(args.name, "challenger")
        print(f"v{v2} >= v{v1}: promoted v{v2} to 'champion', archived v{v1}.")
    else:
        print(f"v{v2} < v{v1}: kept v{v1} as champion (v{v2} stays challenger).")

    champ = client.get_model_version_by_alias(args.name, "champion")
    aliases = client.get_registered_model(args.name).aliases  # {alias: version}
    print(f"\nfinal champion: {args.name} v{champ.version}")
    print(f"aliases: {aliases}")
    print("all versions:")
    versions = client.search_model_versions(f"name='{args.name}'")
    for mv in sorted(versions, key=lambda m: int(m.version)):
        alias = [a for a, v in aliases.items() if v == mv.version]
        print(f"  v{mv.version}  aliases={alias}  tags={dict(mv.tags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
