def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "test:0.1.0"


def test_score_contract(client):
    txn = {
        "transaction_id": "t1",
        "transaction_amt": 980.0,
        "transaction_dt": "2024-05-01T03:30:00Z",
        "product_cd": "C",
        "purchaser_email_domain": "anonymous.com",
        "C1": 9,
        "C13": 14,
    }
    r = client.post("/score", json=txn)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "t1"
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["fraud_decision"] in {"approve", "review", "decline"}
    assert len(body["top_3_reasons"]) == 3
    assert body["model_version"] == "test:0.1.0"
    reason = body["top_3_reasons"][0]
    assert set(reason) == {"feature", "value", "impact", "direction"}


def test_score_batch(client):
    batch = {"transactions": [
        {"transaction_amt": 980.0, "product_cd": "C", "purchaser_email_domain": "anonymous.com"},
        {"transaction_amt": 12.5, "product_cd": "W", "purchaser_email_domain": "gmail.com"},
    ]}
    r = client.post("/score_batch", json=batch)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_negative_amount_rejected(client):
    r = client.post("/score", json={"transaction_amt": -1.0})
    assert r.status_code == 422


def test_empty_batch_rejected(client):
    r = client.post("/score_batch", json={"transactions": []})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    client.post("/score", json={"transaction_amt": 100.0})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "fraud_scores_total" in r.text
