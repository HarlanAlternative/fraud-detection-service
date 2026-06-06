"""Kafka consumer that scores the transaction stream in real time.

Consumes the ``transactions`` topic, scores each event with the same :class:`FraudScorer`
the REST API uses, updates the Prometheus series, and logs the inference to Postgres. Exposes
its own ``/metrics`` on :8001 so Prometheus can scrape the streaming path separately.

    python -m streaming.consumer
"""

from __future__ import annotations

import argparse
import json

from prometheus_client import start_http_server

from fraud.config import get_settings
from fraud.features.engineering import prepare_serving_frame
from fraud.serving import metrics as M
from fraud.serving.db import InferenceLogger
from fraud.serving.scorer import FraudScorer


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Score the Kafka transaction stream.")
    parser.add_argument("--bootstrap", default=settings.kafka_bootstrap)
    parser.add_argument("--topic", default=settings.kafka_topic)
    parser.add_argument("--group", default=settings.kafka_group)
    parser.add_argument("--metrics-port", type=int, default=8001)
    args = parser.parse_args(argv)

    from confluent_kafka import Consumer

    scorer = FraudScorer.from_path()
    M.MODEL_LOADED.set(1)
    logger = InferenceLogger()
    start_http_server(args.metrics_port)
    print(f"[consumer] model={scorer.model_version}; metrics on :{args.metrics_port}")

    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap,
            "group.id": args.group,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([args.topic])
    print(f"[consumer] subscribed to '{args.topic}' @ {args.bootstrap}")

    n = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[consumer] error: {msg.error()}")
                continue
            record = json.loads(msg.value())
            with M.SCORING_LATENCY.time():
                result = scorer.score_one(record)
            result["transaction_id"] = record.get("transaction_id")
            M.observe_result(result)
            # log engineered features (match the drift reference, not raw request values)
            logger.log(prepare_serving_frame(record).iloc[0].to_dict(), result)
            n += 1
            if n % 200 == 0:
                print(f"[consumer] scored {n} (last decision={result['fraud_decision']})")
    except KeyboardInterrupt:
        print(f"[consumer] stopping after {n} messages")
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
