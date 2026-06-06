# Data

Raw datasets are **not committed** (see [.gitignore](../.gitignore)). Download them with the
Kaggle CLI, then place the extracted files under `data/raw/<dataset>/`.

```
data/
├─ raw/                # downloaded, untouched
│  ├─ ieee-cis/        # train_transaction.csv, train_identity.csv, test_*.csv
│  ├─ credit-card/     # creditcard.csv
│  └─ paysim/          # PS_20174392719_1491204439457_log.csv
├─ interim/            # joined / cleaned intermediate parquet
└─ processed/          # model-ready feature matrices
```

## Prerequisites

1. A Kaggle account.
2. `pip install kaggle` (already in `pyproject.toml`).
3. API token: Kaggle → *Account* → *Create New API Token* → save `kaggle.json` to
   `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Unix).
4. **IEEE-CIS only:** open the
   [competition page](https://www.kaggle.com/c/ieee-fraud-detection) once and click
   *Join Competition* / accept the rules, or the download returns 403.

## Download

```powershell
# from the repo root, with the venv active
python -m fraud.data.download --dataset ieee-cis      # primary training set (~590K rows)
python -m fraud.data.download --dataset credit-card   # lightweight baseline + test fixture
python -m fraud.data.download --dataset paysim        # drifted-stream generator (optional)
python -m fraud.data.download --all
```

| Dataset | Kaggle ref | Type | Use in this project |
|---|---|---|---|
| IEEE-CIS | `c/ieee-fraud-detection` (competition) | real Vesta data, multi-table | **primary training set** |
| Credit Card Fraud | `mlg-ulb/creditcardfraud` | 284K PCA-anonymised | baseline + unit-test fixture |
| PaySim | `ealaxi/paysim1` | 6.3M synthetic mobile payments | drifted stream for drift alerts |

> The download is a **manual prerequisite** — nothing else in the pipeline fetches data on
> its own. After downloading, verify with `python -m fraud.data.load --check`.
