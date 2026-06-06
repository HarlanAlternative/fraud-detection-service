# Model Card — Real-time Fraud Detection Ensemble

## Overview
A stacked ensemble that scores card transactions for fraud and returns a calibrated score, an
approve/review/decline decision, and SHAP-based reason codes. Built as a portfolio project on
public benchmark data; **not a deployed production bank system**.

| | |
|---|---|
| **Model** | Stacking: XGBoost + Isolation Forest + Autoencoder → logistic-regression meta-learner |
| **Input** | One transaction over a curated, interpretable feature set (~20 features) |
| **Output** | `fraud_score ∈ [0,1]`, `fraud_decision ∈ {approve, review, decline}`, `top_3_reasons`, `model_version` |
| **Training data** | IEEE-CIS Fraud Detection (primary); Kaggle Credit Card Fraud (baseline); synthetic generator for offline/CI |
| **Owner** | Harlan Li |

## Intended use
- **In scope:** real-time transaction fraud scoring with a human review queue; an example of
  explainable, monitored tabular ML.
- **Out of scope:** AML case management, identity verification, or any automated adverse action
  without human review. Decisions are advisory inputs to a fraud-ops workflow.

## Why three models
- **XGBoost** — supervised workhorse; captures the non-linear interaction patterns of known fraud.
- **Isolation Forest** — unsupervised; flags out-of-distribution ("zero-day") transactions.
- **Autoencoder** — deep reconstruction-error signal on "normal" traffic.
- **LR meta-learner** — combines the three; weights are inspectable (typically XGBoost-dominant).

## Performance
Headline metric is **recall at false-positive-rate = 2%** plus AUC / average precision, measured
on a **chronologically held-out test slice** (time-aware split avoids look-ahead leakage). The
stacked ensemble is the best or tied-best model on AUC and recall@FPR=2%; full five-model table is
in [`notebooks/03_benchmark.ipynb`](../notebooks/03_benchmark.ipynb).

> Absolute numbers depend on the dataset. On the synthetic generator (used for offline runs/CI)
> the pipeline reaches AUC ≈ 0.80; on the real IEEE-CIS features the same pipeline scores
> materially higher. **Production thresholds are calibrated against industry FPR targets**, not
> claimed from a real deployment.

## Decision policy
Two thresholds tuned on a holdout slice encode both business constraints:
- **decline** at the FPR budget (≤ 2%) — only highest-confidence fraud is auto-declined;
- **review** at the target-recall operating point (0.90) — the uncertain middle goes to a human
  queue, so decline+review together reach the recall target;
- everything below is **approved**.

## Class imbalance
Fraud is ~0.2–4% of rows. Handled with `scale_pos_weight` (class weights) by default; undersampling
and SMOTE are compared in the benchmark. **SMOTE is applied only inside the training fold** (via an
`imblearn` pipeline) to avoid leakage.

## Explainability
`top_3_reasons` come from a `TreeExplainer` on the XGBoost base learner, with one-hot SHAP
contributions aggregated back to the original curated feature so the codes are human-readable
(e.g. `purchaser_email_domain=anonymous (increases_risk)`).

## Fairness
The benchmark data has no demographics, so a sub-group audit attaches **synthetic** age/region
proxies and reports decline rate, FPR, and recall per group plus a worst/best **disparity ratio**
(≈ 1.0 = equitable; > 1.2 → review). See [`notebooks/04_fairness.ipynb`](../notebooks/04_fairness.ipynb).
NZ Privacy Act 2020 and Human Rights Act make this audit a deployment prerequisite.

## Monitoring & retraining
- **Drift:** per-feature PSI and prediction-score PSI vs the training reference (alert at PSI > 0.2),
  plus P95 latency (SLA < 100 ms) and rolling **backfill recall** as fraud labels arrive late.
- **Retrain triggers:** significant feature/score drift, or backfill recall dropping below 0.80.
- **Promotion:** retrained candidates are registered in MLflow and promoted via a champion/challenger
  gate (see `fraud.train.registry_demo`).

## Limitations
- Trained on public/synthetic data; real fraud distributions, adversarial adaptation, and label
  delay differ.
- The curated feature set trades a little raw accuracy for interpretability (no anonymous `V*`
  columns), by design.
- Unsupervised members add most value under drift/novel fraud, less so on a static test split.
