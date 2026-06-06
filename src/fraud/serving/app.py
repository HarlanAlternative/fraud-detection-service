"""FastAPI scoring service.

Mirrors the structure of the team's other model microservices: a ``lifespan`` that loads the
model once, a ``/health`` probe, and structured endpoints. Adds pandera input validation,
Prometheus instrumentation, best-effort inference logging, and a label-backfill endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandera.errors as pa_errors
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraud.data.schema import scoring_input_schema
from fraud.features.engineering import prepare_serving_frame
from fraud.serving import metrics as M
from fraud.serving.db import InferenceLogger
from fraud.serving.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ScoreResponse,
    TransactionRequest,
)
from fraud.serving.scorer import FraudScorer


class State:
    scorer: FraudScorer | None = None
    logger: InferenceLogger | None = None


state = State()
_INPUT_SCHEMA = scoring_input_schema()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.scorer = FraudScorer.from_path()
        M.MODEL_LOADED.set(1)
        print(f"[startup] model loaded: {state.scorer.model_version}")
    except FileNotFoundError as exc:
        M.MODEL_LOADED.set(0)
        print(f"[startup] no model available: {exc}")
    state.logger = InferenceLogger()
    yield


app = FastAPI(title="Fraud Detection Scoring Service", version="1.0.0", lifespan=lifespan)


def _require_scorer() -> FraudScorer:
    if state.scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train one first.")
    return state.scorer


def _score(requests: list[TransactionRequest]) -> list[ScoreResponse]:
    scorer = _require_scorer()
    records = [r.model_dump() for r in requests]
    frame = prepare_serving_frame(records)
    try:
        _INPUT_SCHEMA.validate(frame, lazy=True)
    except pa_errors.SchemaErrors as exc:
        raise HTTPException(status_code=422, detail=f"input validation failed: {exc}") from exc

    with M.SCORING_LATENCY.time():
        results = scorer.score_frame(frame)

    responses = []
    for i, (req, res) in enumerate(zip(requests, results, strict=True)):
        res["transaction_id"] = req.transaction_id
        M.observe_result(res)
        if state.logger:
            # log the engineered features the model scored (so drift compares like-for-like
            # with the training reference — raw request values aren't normalised)
            state.logger.log(frame.iloc[i].to_dict(), res)
        responses.append(ScoreResponse(**res))
    return responses


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = state.scorer is not None
    M.REQUESTS_TOTAL.labels(endpoint="health", status="ok").inc()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=state.scorer.model_version if loaded else None,
    )


@app.post("/score", response_model=ScoreResponse)
async def score(request: TransactionRequest) -> ScoreResponse:
    result = _score([request])[0]
    M.REQUESTS_TOTAL.labels(endpoint="score", status="ok").inc()
    return result


@app.post("/score_batch", response_model=BatchResponse)
async def score_batch(request: BatchRequest) -> BatchResponse:
    if not request.transactions:
        raise HTTPException(status_code=422, detail="transactions must not be empty")
    results = _score(request.transactions)
    M.REQUESTS_TOTAL.labels(endpoint="score_batch", status="ok").inc()
    return BatchResponse(results=results)


@app.post("/backfill")
async def backfill(transaction_id: str, label: int) -> dict:
    """Attach a late-arriving ground-truth label to a previously scored transaction."""
    if label not in (0, 1):
        raise HTTPException(status_code=422, detail="label must be 0 or 1")
    updated = state.logger.backfill_label(transaction_id, label) if state.logger else 0
    return {"transaction_id": transaction_id, "label": label, "rows_updated": updated}


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main() -> None:
    import uvicorn

    from fraud.config import get_settings

    uvicorn.run(app, host="0.0.0.0", port=get_settings().api_port)


if __name__ == "__main__":
    main()
