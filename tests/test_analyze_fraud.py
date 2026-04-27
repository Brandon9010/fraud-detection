import pandas as pd
import pytest
from analyze_fraud import score_transactions, summarize_results


def _transactions():
    return pd.DataFrame([
        dict(transaction_id=1, account_id=1, timestamp="2026-01-01", amount_usd=50,
             merchant_category="grocery", channel="web", device_risk_score=10,
             ip_country="US", is_international=0, velocity_24h=1,
             failed_logins_24h=0, chargeback_within_60d=0),
        dict(transaction_id=2, account_id=1, timestamp="2026-01-01", amount_usd=1500,
             merchant_category="electronics", channel="web", device_risk_score=80,
             ip_country="PH", is_international=1, velocity_24h=8,
             failed_logins_24h=6, chargeback_within_60d=1),
        dict(transaction_id=3, account_id=1, timestamp="2026-01-01", amount_usd=750,
             merchant_category="travel", channel="mobile", device_risk_score=50,
             ip_country="US", is_international=0, velocity_24h=3,
             failed_logins_24h=2, chargeback_within_60d=0),
    ])


def _accounts():
    return pd.DataFrame([
        dict(account_id=1, customer_name="Test", country="US", signup_date="2022-01-01",
             kyc_level="full", account_age_days=365, prior_chargebacks=0, is_vip="N"),
    ])


def _chargebacks():
    return pd.DataFrame([
        dict(transaction_id=2, chargeback_date="2026-03-01",
             chargeback_reason="card_not_present", loss_amount_usd=1500),
    ])


# ── score_transactions ────────────────────────────────────────────────────────

def test_score_transactions_adds_risk_score_column():
    scored = score_transactions(_transactions(), _accounts())
    assert "risk_score" in scored.columns


def test_score_transactions_adds_risk_label_column():
    scored = score_transactions(_transactions(), _accounts())
    assert "risk_label" in scored.columns


def test_risk_scores_are_in_valid_range():
    scored = score_transactions(_transactions(), _accounts())
    assert scored["risk_score"].between(0, 100).all()


def test_risk_labels_are_valid_values():
    scored = score_transactions(_transactions(), _accounts())
    assert set(scored["risk_label"]).issubset({"low", "medium", "high"})


def test_high_risk_transaction_labelled_high():
    scored = score_transactions(_transactions(), _accounts())
    row = scored[scored["transaction_id"] == 2].iloc[0]
    assert row["risk_label"] == "high"


def test_low_risk_transaction_labelled_low():
    scored = score_transactions(_transactions(), _accounts())
    row = scored[scored["transaction_id"] == 1].iloc[0]
    assert row["risk_label"] == "low"


def test_row_count_unchanged_after_scoring():
    txns = _transactions()
    scored = score_transactions(txns, _accounts())
    assert len(scored) == len(txns)


# ── summarize_results ─────────────────────────────────────────────────────────

def test_summarize_results_has_required_columns():
    scored = score_transactions(_transactions(), _accounts())
    summary = summarize_results(scored, _chargebacks())
    for col in ("risk_label", "transactions", "total_amount_usd", "chargebacks", "chargeback_rate"):
        assert col in summary.columns


def test_transaction_counts_sum_to_total():
    scored = score_transactions(_transactions(), _accounts())
    summary = summarize_results(scored, _chargebacks())
    assert summary["transactions"].sum() == len(_transactions())


def test_chargeback_rate_is_one_for_high_tier():
    scored = score_transactions(_transactions(), _accounts())
    summary = summarize_results(scored, _chargebacks())
    high = summary[summary["risk_label"] == "high"].iloc[0]
    assert high["chargeback_rate"] == 1.0


def test_chargeback_rate_is_zero_for_clean_tier():
    scored = score_transactions(_transactions(), _accounts())
    summary = summarize_results(scored, _chargebacks())
    low = summary[summary["risk_label"] == "low"].iloc[0]
    assert low["chargeback_rate"] == 0.0


def test_total_amount_matches_input():
    scored = score_transactions(_transactions(), _accounts())
    summary = summarize_results(scored, _chargebacks())
    assert summary["total_amount_usd"].sum() == pytest.approx(
        _transactions()["amount_usd"].sum()
    )


def test_no_chargebacks_produces_zero_rate():
    scored = score_transactions(_transactions(), _accounts())
    empty_cbs = pd.DataFrame(columns=["transaction_id"])
    summary = summarize_results(scored, empty_cbs)
    assert (summary["chargeback_rate"] == 0).all()
