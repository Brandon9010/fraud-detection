import pytest
from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    """Minimal transaction that contributes 0 points across every scoring rule."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 50,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ── label_risk ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (0,   "low"),
    (1,   "low"),
    (29,  "low"),
    (30,  "medium"),
    (59,  "medium"),
    (60,  "high"),
    (100, "high"),
])
def test_label_risk_boundaries(score, expected):
    assert label_risk(score) == expected


# ── score bounds ──────────────────────────────────────────────────────────────

def test_base_transaction_scores_zero():
    assert score_transaction(_base_tx()) == 0


def test_score_clamped_at_100():
    # Max possible without clamping: 25+15+25+20+20+20 = 125
    tx = _base_tx(
        device_risk_score=99,
        is_international=1,
        amount_usd=5000,
        velocity_24h=10,
        failed_logins_24h=10,
        prior_chargebacks=5,
    )
    assert score_transaction(tx) == 100


# ── device risk ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("device_score,expected_delta", [
    (0,   0),
    (39,  0),
    (40,  10),
    (69,  10),
    (70,  25),
    (100, 25),
])
def test_device_risk_contribution(device_score, expected_delta):
    assert score_transaction(_base_tx(device_risk_score=device_score)) == expected_delta


# ── international flag ────────────────────────────────────────────────────────

@pytest.mark.parametrize("is_intl,expected_delta", [
    (0, 0),
    (1, 15),
])
def test_international_contribution(is_intl, expected_delta):
    assert score_transaction(_base_tx(is_international=is_intl)) == expected_delta


# ── amount ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("amount,expected_delta", [
    (0,       0),
    (499.99,  0),
    (500,     10),
    (999.99,  10),
    (1000,    25),
    (9999,    25),
])
def test_amount_contribution(amount, expected_delta):
    assert score_transaction(_base_tx(amount_usd=amount)) == expected_delta


# ── velocity ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("velocity,expected_delta", [
    (1,  0),
    (2,  0),
    (3,  5),
    (5,  5),
    (6,  20),
    (20, 20),
])
def test_velocity_contribution(velocity, expected_delta):
    assert score_transaction(_base_tx(velocity_24h=velocity)) == expected_delta


# ── failed logins ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("logins,expected_delta", [
    (0,  0),
    (1,  0),
    (2,  10),
    (4,  10),
    (5,  20),
    (50, 20),
])
def test_failed_logins_contribution(logins, expected_delta):
    assert score_transaction(_base_tx(failed_logins_24h=logins)) == expected_delta


# ── prior chargebacks ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("chargebacks,expected_delta", [
    (0,  0),
    (1,  5),
    (2,  20),
    (10, 20),
])
def test_prior_chargebacks_contribution(chargebacks, expected_delta):
    assert score_transaction(_base_tx(prior_chargebacks=chargebacks)) == expected_delta


# ── additivity ────────────────────────────────────────────────────────────────

def test_independent_signals_add():
    # international (+15) + high velocity (+20) = 35
    assert score_transaction(_base_tx(is_international=1, velocity_24h=8)) == 35


def test_all_mid_tier_signals_add():
    # device 40-69 (+10) + amount 500-999 (+10) + velocity 3-5 (+5)
    # + failed logins 2-4 (+10) + 1 prior CB (+5) = 40
    tx = _base_tx(
        device_risk_score=55,
        amount_usd=750,
        velocity_24h=4,
        failed_logins_24h=3,
        prior_chargebacks=1,
    )
    assert score_transaction(tx) == 40


# ── real-world profiles from dataset ─────────────────────────────────────────

# 8 confirmed chargebacks — none should ever score "low"
@pytest.mark.parametrize("tx_id,tx_attrs,expected_label", [
    ("50003", dict(device_risk_score=81, is_international=1, amount_usd=1250,  velocity_24h=6,  failed_logins_24h=5, prior_chargebacks=0), "high"),
    ("50006", dict(device_risk_score=77, is_international=1, amount_usd=399.99, velocity_24h=7,  failed_logins_24h=6, prior_chargebacks=3), "high"),
    ("50008", dict(device_risk_score=68, is_international=1, amount_usd=620,   velocity_24h=5,  failed_logins_24h=3, prior_chargebacks=0), "medium"),
    ("50011", dict(device_risk_score=85, is_international=1, amount_usd=1400,  velocity_24h=8,  failed_logins_24h=7, prior_chargebacks=1), "high"),
    ("50013", dict(device_risk_score=79, is_international=1, amount_usd=150,   velocity_24h=7,  failed_logins_24h=5, prior_chargebacks=0), "high"),
    ("50014", dict(device_risk_score=72, is_international=1, amount_usd=49.99, velocity_24h=9,  failed_logins_24h=7, prior_chargebacks=3), "high"),
    ("50015", dict(device_risk_score=71, is_international=1, amount_usd=910,   velocity_24h=6,  failed_logins_24h=4, prior_chargebacks=0), "high"),
    ("50019", dict(device_risk_score=83, is_international=1, amount_usd=75,    velocity_24h=10, failed_logins_24h=8, prior_chargebacks=1), "high"),
])
def test_chargeback_transactions_not_low_risk(tx_id, tx_attrs, expected_label):
    result = label_risk(score_transaction(_base_tx(**tx_attrs)))
    assert result == expected_label, f"TX {tx_id}: expected {expected_label}, got {result}"


# Clean domestic profiles should stay low risk
@pytest.mark.parametrize("tx_id,tx_attrs", [
    ("50001", dict(device_risk_score=8,  amount_usd=45.2)),
    ("50004", dict(device_risk_score=12, amount_usd=14.99)),
    ("50009", dict(device_risk_score=6,  amount_usd=18.4)),
    ("50012", dict(device_risk_score=10, amount_usd=64.5)),
    ("50020", dict(device_risk_score=15, amount_usd=120)),
])
def test_clean_domestic_transactions_score_low(tx_id, tx_attrs):
    result = label_risk(score_transaction(_base_tx(**tx_attrs)))
    assert result == "low", f"TX {tx_id}: expected low, got {result}"
