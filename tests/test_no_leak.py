"""Tests for backtest harness: no leakage, valid output."""
import pytest
import pandas as pd
from src.backtest import backtest, TOURNAMENT_PANEL


@pytest.fixture(scope="module")
def bt_wc2018():
    """Run backtest once for WC2018, reuse across tests."""
    return backtest(["WC2018"], verbose=False)


def test_no_leakage(bt_wc2018):
    """Train data must end strictly before tournament start for every match."""
    cutoff = pd.Timestamp(TOURNAMENT_PANEL["WC2018"][1])
    assert len(bt_wc2018) > 0
    for _, row in bt_wc2018.iterrows():
        assert row["date"] >= cutoff, (
            f"LEAK: match date {row['date']} before cutoff {cutoff}"
        )


def test_backtest_has_required_columns(bt_wc2018):
    required = ["tournament", "date", "home", "away", "home_score", "away_score",
                 "outcome", "p_home", "p_draw", "p_away", "rps", "logloss", "brier"]
    for col in required:
        assert col in bt_wc2018.columns, f"Missing column: {col}"


def test_backtest_probabilities_valid(bt_wc2018):
    for _, row in bt_wc2018.iterrows():
        total = row["p_home"] + row["p_draw"] + row["p_away"]
        assert abs(total - 1.0) < 1e-4, f"Probs don't sum to 1: {total}"
        assert row["p_home"] >= 0
        assert row["p_draw"] >= 0
        assert row["p_away"] >= 0


def test_rps_in_valid_range(bt_wc2018):
    """RPS must be in [0, 1] for every match."""
    for _, row in bt_wc2018.iterrows():
        assert 0 <= row["rps"] <= 1.0, f"RPS out of range: {row['rps']}"


def test_match_count_wc2018(bt_wc2018):
    """WC2018 should have 64 matches."""
    assert len(bt_wc2018) == 64, f"Expected 64 matches, got {len(bt_wc2018)}"
