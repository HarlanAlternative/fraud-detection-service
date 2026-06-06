"""Transaction stream producer.

Generates a stream of transaction events (from the synthetic IEEE-CIS generator, or PaySim
if present) and publishes them to the Kafka ``transactions`` topic. ``--drift`` shifts the
feature distribution so the drift monitor and alerts have something to fire on — the live
demonstration that PSI catches distribution shift.

    python -m streaming.producer --n 2000 --rate 50           # normal traffic
    python -m streaming.producer --n 2000 --drift             # drifted traffic
    python -m streaming.producer --n 20 --no-broker           # offline: score locally, no Kafka
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

import pandas as pd

from fraud.config import get_settings
from fraud.data.synthetic import make_ieee_like


def raw_row_to_request(row: pd.Series) -> dict:
    """Map a raw IEEE-CIS-style row to the scoring API's request shape."""
    def email(v):
        return None if pd.isna(v) else str(v)

    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
        "transaction_amt": float(row["TransactionAmt"]),
        "product_cd": str(row["ProductCD"]),
        "card_network": email(row.get("card4")),
        "card_type": email(row.get("card6")),
        "purchaser_email_domain": email(row.get("P_emaildomain")),
        "recipient_email_domain": email(row.get("R_emaildomain")),
        "device_type": email(row.get("DeviceType")),
        "dist1": None if pd.isna(row.get("dist1")) else float(row["dist1"]),
        "card1": float(row["card1"]),
        "card2": None if pd.isna(row.get("card2")) else float(row["card2"]),
        "addr1": None if pd.isna(row.get("addr1")) else float(row["addr1"]),
        "C1": float(row["C1"]), "C2": float(row["C2"]),
        "C13": float(row["C13"]), "C14": float(row["C14"]),
        "D1": float(row["D1"]),
        "D4": None if pd.isna(row.get("D4")) else float(row["D4"]),
        "D15": float(row["D15"]),
    }


def transaction_stream(n: int, *, drift: bool = False, seed: int = 123):
    """Yield ``n`` transaction request dicts."""
    df = make_ieee_like(n, drift=drift, seed=seed).drop(columns=["isFraud"])
    for _, row in df.iterrows():
        yield raw_row_to_request(row)


def _run_no_broker(n: int, drift: bool) -> int:
    """Offline mode: score locally and print the decision mix (no Kafka required)."""
    from collections import Counter

    from fraud.serving.scorer import FraudScorer

    scorer = FraudScorer.from_path()
    decisions = Counter()
    for rec in transaction_stream(n, drift=drift):
        decisions[scorer.score_one(rec)["fraud_decision"]] += 1
    print(f"scored {n} ({'drifted' if drift else 'normal'}) transactions: {dict(decisions)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Publish a transaction stream to Kafka.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=50.0, help="messages per second")
    parser.add_argument("--drift", action="store_true", help="emit a drifted distribution")
    parser.add_argument("--bootstrap", default=settings.kafka_bootstrap)
    parser.add_argument("--topic", default=settings.kafka_topic)
    parser.add_argument("--no-broker", action="store_true", help="score locally, no Kafka")
    args = parser.parse_args(argv)

    if args.no_broker:
        return _run_no_broker(args.n, args.drift)

    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": args.bootstrap})
    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    sent = 0
    for rec in transaction_stream(args.n, drift=args.drift):
        producer.produce(args.topic, key=rec["transaction_id"], value=json.dumps(rec))
        producer.poll(0)
        sent += 1
        if sent % 200 == 0:
            producer.flush()
            print(f"  produced {sent}/{args.n}")
        if delay:
            time.sleep(delay)
    producer.flush()
    print(f"done: produced {sent} messages to '{args.topic}' (drift={args.drift})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
