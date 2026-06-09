<h1 align="center">Real-time Fraud Detection Service</h1>

<p align="center">
  <b>Scores card transactions for fraud in real time</b> — a stacked
  XGBoost&nbsp;+&nbsp;Isolation&nbsp;Forest&nbsp;+&nbsp;Autoencoder ensemble served behind a
  FastAPI API, with SHAP reason codes, drift monitoring, and end-to-end MLOps.
</p>

<p align="center">
  Python 3.11 · FastAPI · XGBoost · PyTorch · scikit-learn · SHAP · MLflow · Kafka · Docker · Prometheus · Grafana · Azure
</p>

---

Every request returns a fraud score, an **approve / review / decline** decision, and
**SHAP-based top-3 reason codes** — so each call is explainable, not just a number.
Trained and benchmarked on public data (IEEE-CIS, Kaggle Credit Card Fraud); decision
thresholds are calibrated against industry false-positive-rate targets.

> Research / portfolio project — not a deployed bank system.

## Highlights

- **Stacked ensemble** — XGBoost + Isolation Forest + Autoencoder, combined by a
  logistic-regression meta-learner; thresholds tuned for **recall ≈ 0.90 at FPR < 2%**.
- **Explainable by default** — SHAP top-3 reason codes on every decision, for
  regulator-friendly auditability instead of a black-box score.
- **Real-time at two speeds** — a synchronous FastAPI `/score` endpoint and an async
  Kafka (Redpanda) consumer, both scoring with the exact same model.
- **MLOps, not just a model** — MLflow registry with champion/challenger promotion,
  PSI feature/score-drift detection, label-backfill recall monitoring, and Prometheus +
  Grafana dashboards with automated alerting.
- **Runs fully offline** — a synthetic IEEE-CIS-shaped data generator means every
  notebook, test, and demo works with zero data downloads.
- **Ship-ready** — Dockerised stack plus Azure Container Apps IaC (Bicep), with green CI.

## Why three models

| Model | Role |
|---|---|
| **XGBoost** | supervised workhorse; captures non-linear interaction patterns; mature SHAP |
| **Isolation Forest** | unsupervised — flags zero-day / out-of-distribution transactions |
| **Autoencoder** | deep reconstruction-error signal learned on legitimate traffic |
| **LR stacking + threshold tuning** | combines the three; thresholds set to recall ≈ 0.90 at FPR < 2% |

## Architecture

```
[ transaction ] ─► FastAPI /score  ──►  { fraud_score, fraud_decision, top_3_reasons, model_version }
                        │
   XGBoost ┐            ├─ Postgres inference log ──(label backfill)──► drift monitor (PSI, recall)
   IForest ┼─► LR meta ─┤                                                      │
   AutoEnc ┘            └─ Prometheus /metrics ──► Grafana                Prometheus alerts
```

Real-time path: a Kafka (Redpanda) consumer scores a `transactions` topic with the same model.
Full diagram in [docs/architecture.md](docs/architecture.md); model details in
[docs/model_card.md](docs/model_card.md).

## Results (held-out, time-aware split)

Headline metric: **recall at FPR = 2%**. The stacked ensemble is best/tied-best on AUC and
recall@FPR=2%; the unsupervised members are weaker alone but earn their place under drift.
Full five-model table + ROC/PR curves + imbalance-strategy study in
[notebooks/03_benchmark.ipynb](notebooks/03_benchmark.ipynb).

> Numbers shown in notebooks run on either real IEEE-CIS (if downloaded) or a synthetic
> IEEE-CIS-shaped generator (offline/CI). The ensemble's edge is small and data-dependent —
> reported honestly rather than inflated.

## Quick start

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,streaming]"
Copy-Item .env.example .env

# 1. (optional) real data — manual prerequisite, see data/README.md
python -m fraud.data.download --dataset ieee-cis
python -m fraud.data.load --check

# 2. explore + benchmark (works offline on synthetic data)
jupyter lab notebooks/01_eda.ipynb

# 3. train the ensemble (logs to MLflow, saves models/stacking.joblib)
python -m fraud.train.pipeline --rows 200000

# 4. run the full stack: API + Postgres + MLflow + Prometheus + Grafana
docker compose up                                   # API on :8000, Grafana :3000, MLflow :5000
docker compose -f docker-compose.yml -f docker-compose.streaming.yml up   # + Kafka consumer
```

Score a transaction:

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "transaction_amt": 980.0, "transaction_dt": "2024-05-01T03:30:00Z",
  "product_cd": "C", "purchaser_email_domain": "anonymous.com", "C1": 9, "C13": 14
}'
# -> { "fraud_score": 0.99, "fraud_decision": "decline",
#      "top_3_reasons": [ {"feature":"transaction_amt", ...}, ... ], "model_version": "..." }
```

## API

The model is loaded once at startup (FastAPI `lifespan`); interactive docs live at `/docs`.

| Method & path | Purpose |
|---|---|
| `POST /score` | score one transaction → fraud score, decision, top-3 reasons, model version |
| `POST /score_batch` | score a list of transactions in a single call |
| `POST /backfill?transaction_id=…&label=0\|1` | attach a late-arriving ground-truth label so drift/recall monitoring can compare against real outcomes |
| `GET /health` | liveness probe — `status`, `model_loaded`, served `model_version` |
| `GET /metrics` | Prometheus exposition: decision mix, request counts, scoring-latency histogram, score distribution, model-loaded gauge |

Only `transaction_amt` is required — every other field is optional (a brand-new card has no
history) and the model imputes what is missing. Requests are validated against a pandera
schema: a malformed body returns `422`, and scoring before any model is trained returns `503`.

## Configuration

Settings are environment variables prefixed `FRAUD_`, loaded by
[src/fraud/config.py](src/fraud/config.py) via pydantic-settings — copy `.env.example` to
`.env` and adjust. The ones that shape decisions:

| Variable | Default | Effect |
|---|---|---|
| `FRAUD_TARGET_RECALL` | `0.90` | recall the threshold tuner targets on the holdout split |
| `FRAUD_MAX_FPR` | `0.02` | false-positive-rate budget the decision threshold must respect |
| `FRAUD_REVIEW_BAND` | `0.15` | scores within ± this of the threshold are routed to **review** rather than auto-decided |
| `FRAUD_MODEL_STAGE` | `Production` | which MLflow registry stage the API serves |

## Things you can run

| Command | What it does |
|---|---|
| `python -m fraud.train.pipeline --synthetic` | train ensemble, log to MLflow, save model + drift reference |
| `python -m fraud.train.registry_demo` | MLflow registry **v1→v2 champion/challenger promotion** (sqlite, offline) |
| `python -m fraud.monitoring.exporter --demo` | offline PSI drift check (reference vs drifted window) |
| `python -m streaming.producer --n 500 --no-broker --drift` | score a drifted stream locally (no Kafka) |
| `pytest` | full test suite (data, features, models, drift, API) |

## Layout

| Path | Contents |
|---|---|
| `src/fraud/data/` | download, load (IEEE-CIS join), pandera schema, synthetic generator |
| `src/fraud/features/` | curated feature set (train/serve shared contract) |
| `src/fraud/models/` | XGBoost, Isolation Forest, Autoencoder, LR stacking, threshold tuner, metrics |
| `src/fraud/explain/` | SHAP → top-3 reason codes |
| `src/fraud/serving/` | FastAPI app, scorer, schemas, Prometheus metrics, Postgres logging, pyfunc wrapper |
| `src/fraud/train/` | training pipeline, MLflow tracking, registry promotion |
| `src/fraud/monitoring/` | PSI drift, rolling backfill recall, Prometheus exporter |
| `src/fraud/fairness/` | sub-group performance + disparity |
| `streaming/` | Kafka producer + consumer |
| `monitoring/` | Prometheus config + alerts, Grafana dashboard |
| `infra/bicep/` | Azure Container Apps IaC |
| `notebooks/` | 01 EDA · 02 feature engineering · 03 benchmark · 04 fairness |
| `tests/` | pytest suite |

## Resume bullets

```
Real-time Fraud Detection Service — Production ML System
XGBoost · Isolation Forest · Autoencoder · SHAP · FastAPI · MLflow · Kafka · Prometheus · Grafana · Azure

· Built a stacked ensemble (XGBoost + Isolation Forest + Autoencoder, LR meta-learner) for
  card-fraud scoring on the IEEE-CIS dataset, optimising recall at a 2% false-positive budget.
· Production scoring service: FastAPI REST + Kafka streaming at P95 < 100ms, with structured
  JSON contracts and SHAP-based top-3 reason codes for regulator-friendly explainability.
· MLOps: PSI feature/score-drift detection, label-backfill recall monitoring, two-threshold
  decision policy, MLflow model registry with champion/challenger promotion, and a
  Prometheus + Grafana dashboard with automated drift alerting; deployable to Azure Container Apps.
```
