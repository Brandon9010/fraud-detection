from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
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


def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) >= 25


def test_high_device_risk_adds_risk():
    low = score_transaction(_base_tx(device_risk_score=10))
    high = score_transaction(_base_tx(device_risk_score=75))
    assert high > low


def test_international_adds_risk():
    domestic = score_transaction(_base_tx(is_international=0))
    international = score_transaction(_base_tx(is_international=1))
    assert international > domestic


def test_high_velocity_adds_risk():
    low_v = score_transaction(_base_tx(velocity_24h=1))
    high_v = score_transaction(_base_tx(velocity_24h=8))
    assert high_v > low_v


def test_prior_chargebacks_add_risk():
    clean = score_transaction(_base_tx(prior_chargebacks=0))
    one = score_transaction(_base_tx(prior_chargebacks=1))
    repeat = score_transaction(_base_tx(prior_chargebacks=3))
    assert one > clean
    assert repeat > one


def test_confirmed_fraud_profile_scores_high():
    # Mirrors TX 50011: device_risk=85, international, $1400, velocity=8, logins=7, 1 prior CB
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1400,
        velocity_24h=8,
        failed_logins_24h=7,
        prior_chargebacks=1,
    )
    assert label_risk(score_transaction(tx)) == "high"


def test_clean_low_risk_profile_scores_low():
    # Mirrors TX 50001: low device risk, domestic, small amount, single transaction
    tx = _base_tx(device_risk_score=8, amount_usd=45)
    assert label_risk(score_transaction(tx)) == "low"
