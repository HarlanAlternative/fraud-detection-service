# Architecture

## Training (offline)

```
raw IEEE-CIS (or synthetic)            fraud.data.load / synthetic
        │
        ▼  fraud.features.engineering  (raw → curated, time features)
   curated feature matrix
        │  time-aware split (train / holdout / test)
        ▼  fraud.train.pipeline
  ┌─ XGBoost ─────────→ p_xgb ┐
  ├─ Isolation Forest → anom  │  base learners fit on TRAIN
  └─ Autoencoder ─────→ recon ┘
        │  base scores on HOLDOUT
        ▼
   LR meta-learner  ── fit on holdout ──►  fraud_score
        │
   threshold tuner (decline @ FPR≤2%, review @ recall=0.90)
        │
   MLflow: params + metrics + thresholds + model (pyfunc) → registry
```

Leakage controls: chronological split; base learners never see the holdout the meta-learner is
fit on; the test slice is untouched until final evaluation. SMOTE (when used) lives inside the
training fold only.

## Serving (online)

```
                     ┌──────────────── Prometheus ──── Grafana
                     │  scrape /metrics
HTTP  ─► FastAPI ────┤
 or                  ├─► FraudScorer (model + SHAP) ─► {score, decision, top_3_reasons}
Kafka ─► consumer ───┤
                     └─► Postgres inference_log ──(label backfill)──► drift monitor
                                                                          │ PSI / recall gauges
                                                                          ▼  Prometheus alerts
```

- **fraud.serving.app** — REST `/score`, `/score_batch`, `/health`, `/metrics`, `/backfill`.
- **streaming.consumer** — same `FraudScorer` over a Kafka `transactions` topic.
- **fraud.serving.db** — best-effort inference logging + late ground-truth backfill.
- **fraud.monitoring.exporter** — PSI (feature + score) and rolling backfill recall → Prometheus.

## Components

| Layer | Module | Notes |
|---|---|---|
| Config | `fraud.config` | env-driven (`FRAUD_*`), one source of truth |
| Data | `fraud.data.{download,load,schema,synthetic}` | Kaggle fetch, IEEE join, pandera, offline generator |
| Features | `fraud.features.engineering` | shared train/serve contract (20 curated features) |
| Models | `fraud.models.{preprocess,xgb,iforest,autoencoder,stacking,threshold,metrics}` | |
| Explain | `fraud.explain.shap_reasons` | XGBoost SHAP → curated-feature reason codes |
| Serving | `fraud.serving.{app,scorer,schemas,metrics,db,model_wrapper}` | FastAPI + pyfunc |
| Streaming | `streaming.{producer,consumer}` | Redpanda (Kafka API) |
| Monitoring | `fraud.monitoring.{drift,exporter}` | PSI + recall |
| Infra | `docker/`, `docker-compose*.yml`, `infra/bicep/` | local stack + Azure Container Apps |

## Local stack

`docker compose up` → Postgres, MLflow, scoring-api, drift-monitor, Prometheus (`:9090`),
Grafana (`:3000`). Add `-f docker-compose.streaming.yml` for Redpanda + the streaming consumer.
