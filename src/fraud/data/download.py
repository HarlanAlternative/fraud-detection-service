"""Download the fraud datasets from Kaggle into ``data/raw/<dataset>/``.

This is the only component that fetches data, and it is a **manual prerequisite** (it
needs Kaggle API credentials and, for IEEE-CIS, accepting the competition rules — see
``data/README.md``). Usage::

    python -m fraud.data.download --dataset ieee-cis
    python -m fraud.data.download --all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from fraud.config import get_settings


@dataclass(frozen=True)
class DatasetSpec:
    """How to fetch a dataset and how to recognise it once downloaded."""

    key: str
    kind: str  # "competition" | "dataset"
    ref: str
    sentinel: str  # a filename that must exist after a successful download


DATASETS: dict[str, DatasetSpec] = {
    "ieee-cis": DatasetSpec(
        key="ieee-cis",
        kind="competition",
        ref="ieee-fraud-detection",
        sentinel="train_transaction.csv",
    ),
    "credit-card": DatasetSpec(
        key="credit-card",
        kind="dataset",
        ref="mlg-ulb/creditcardfraud",
        sentinel="creditcard.csv",
    ),
    "paysim": DatasetSpec(
        key="paysim",
        kind="dataset",
        ref="ealaxi/paysim1",
        sentinel="PS_20174392719_1491204439457_log.csv",
    ),
}


def _authenticated_api():
    """Return an authenticated Kaggle API client, with a friendly error if creds are missing."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise SystemExit(
            "The 'kaggle' package is not installed. Run: pip install -e ."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - surface any auth problem clearly
        raise SystemExit(
            "Kaggle authentication failed. Place kaggle.json in ~/.kaggle/ "
            "(see data/README.md). Original error: " f"{exc}"
        ) from exc
    return api


def download(spec: DatasetSpec, *, force: bool = False) -> None:
    """Download and unzip one dataset into ``data/raw/<key>/``."""
    settings = get_settings()
    dest = settings.raw_dir / spec.key
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / spec.sentinel).exists() and not force:
        print(f"[skip] {spec.key}: already present ({spec.sentinel}). Use --force to redo.")
        return

    api = _authenticated_api()
    print(f"[download] {spec.key} ({spec.kind}:{spec.ref}) -> {dest}")
    try:
        if spec.kind == "competition":
            api.competition_download_files(spec.ref, path=str(dest), quiet=False)
        else:
            api.dataset_download_files(spec.ref, path=str(dest), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if spec.kind == "competition":
            hint = (
                " For IEEE-CIS you must accept the competition rules once at "
                "https://www.kaggle.com/c/ieee-fraud-detection before downloading."
            )
        raise SystemExit(f"Download failed for {spec.key}: {exc}.{hint}") from exc

    # Competition downloads arrive as a single zip; unzip it in place.
    if spec.kind == "competition":
        _unzip_all(dest)

    if not (dest / spec.sentinel).exists():
        print(
            f"[warn] {spec.key}: expected '{spec.sentinel}' not found after download; "
            "inspect the folder manually."
        )
    else:
        print(f"[ok] {spec.key} ready.")


def _unzip_all(folder) -> None:
    """Unzip every .zip in ``folder`` (competition payloads are zipped)."""
    import zipfile

    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
        archive.unlink()  # save disk; the extracted CSVs are what we need


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download fraud datasets from Kaggle.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=sorted(DATASETS), help="single dataset to fetch")
    group.add_argument("--all", action="store_true", help="fetch every dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    specs = DATASETS.values() if args.all else [DATASETS[args.dataset]]
    for spec in specs:
        download(spec, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
