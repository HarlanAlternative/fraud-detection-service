"""Pydantic request/response models — the public API contract.

Field names mirror the curated feature names so a request body flows straight into
``prepare_serving_frame``. Only ``transaction_amt`` is required; everything else is optional
(a brand-new card legitimately has no history), and the model imputes what is missing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str | None = Field(default=None, description="caller-side id (echoed back)")
    transaction_amt: float = Field(..., ge=0, description="transaction amount")
    transaction_dt: str | None = Field(
        default=None, description="ISO-8601 timestamp; hour/day_of_week derived from it"
    )
    # interpretable categoricals
    product_cd: str | None = None
    card_network: str | None = Field(default=None, description="visa / mastercard / amex / ...")
    card_type: str | None = Field(default=None, description="debit / credit")
    purchaser_email_domain: str | None = None
    recipient_email_domain: str | None = None
    device_type: str | None = Field(default=None, description="desktop / mobile")
    # numeric signals (history aggregates / identifiers)
    dist1: float | None = None
    card1: float | None = None
    card2: float | None = None
    addr1: float | None = None
    C1: float | None = None
    C2: float | None = None
    C13: float | None = None
    C14: float | None = None
    D1: float | None = None
    D4: float | None = None
    D15: float | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_001",
                "transaction_amt": 980.0,
                "transaction_dt": "2024-05-01T03:30:00Z",
                "product_cd": "C",
                "card_network": "visa",
                "card_type": "credit",
                "purchaser_email_domain": "anonymous.com",
                "device_type": "mobile",
                "C1": 9,
                "C13": 14,
            }
        }
    }


class ReasonCode(BaseModel):
    feature: str
    value: Any | None = None
    impact: float
    direction: str  # increases_risk | decreases_risk


class ScoreResponse(BaseModel):
    transaction_id: str | None = None
    fraud_score: float
    fraud_decision: str  # approve | review | decline
    top_3_reasons: list[ReasonCode]
    model_version: str


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchResponse(BaseModel):
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
