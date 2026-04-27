import pandas as pd
import pytest
from features import build_model_frame


def _txns(*rows):
    defaults = dict(
        transaction_id=1,
        account_id=1,
        timestamp="2026-01-01",
        amount_usd=100,
        merchant_category="grocery",
        channel="web",
        device_risk_score=10,
        ip_country="US",
        is_international=0,
        velocity_24h=1,
        failed_logins_24h=0,
        chargeback_within_60d=0,
    )
    if not rows:
        rows = ({},)
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _accts(*rows):
    defaults = dict(
        account_id=1,
        customer_name="Test User",
        country="US",
        signup_date="2022-01-01",
        kyc_level="full",
        account_age_days=365,
        prior_chargebacks=0,
        is_vip="N",
    )
    if not rows:
        rows = ({},)
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ── is_large_amount ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("amount,expected_flag", [
    (0,       0),
    (999.99,  0),
    (1000,    1),
    (5000,    1),
])
def test_is_large_amount_flag(amount, expected_flag):
    df = build_model_frame(_txns(dict(amount_usd=amount)), _accts())
    assert df["is_large_amount"].iloc[0] == expected_flag


# ── login_pressure ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("logins,expected_pressure", [
    (0,  "none"),
    (1,  "low"),
    (2,  "low"),
    (3,  "high"),
    (10, "high"),
])
def test_login_pressure_bins(logins, expected_pressure):
    df = build_model_frame(_txns(dict(failed_logins_24h=logins)), _accts())
    assert df["login_pressure"].iloc[0] == expected_pressure


# ── merge behaviour ───────────────────────────────────────────────────────────

def test_all_transactions_kept_when_account_missing():
    txns = _txns(dict(account_id=1), dict(account_id=999))
    accts = _accts(dict(account_id=1))
    df = build_model_frame(txns, accts)
    assert len(df) == 2


def test_account_fields_joined_correctly():
    txns = _txns(dict(account_id=7))
    accts = _accts(dict(account_id=7, prior_chargebacks=3, country="NG", kyc_level="basic"))
    df = build_model_frame(txns, accts)
    assert df["prior_chargebacks"].iloc[0] == 3
    assert df["country"].iloc[0] == "NG"
    assert df["kyc_level"].iloc[0] == "basic"


def test_multiple_transactions_same_account_all_get_account_data():
    txns = _txns(
        dict(transaction_id=1, account_id=1, amount_usd=100),
        dict(transaction_id=2, account_id=1, amount_usd=200),
    )
    accts = _accts(dict(account_id=1, prior_chargebacks=2))
    df = build_model_frame(txns, accts)
    assert len(df) == 2
    assert (df["prior_chargebacks"] == 2).all()


def test_output_contains_engineered_columns():
    df = build_model_frame(_txns(), _accts())
    assert "is_large_amount" in df.columns
    assert "login_pressure" in df.columns


def test_original_transaction_columns_preserved():
    df = build_model_frame(_txns(), _accts())
    for col in ("transaction_id", "amount_usd", "device_risk_score", "is_international"):
        assert col in df.columns
