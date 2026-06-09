from src.backtest import backtest


def test_no_leakage():
    """Verify that for every prediction, train data ends before test data starts."""
    results = backtest(["WC2018"])
    assert len(results) > 0
    for _, row in results.iterrows():
        assert row["train_cutoff"] <= row["match_date"], (
            f"LEAK: train_cutoff {row['train_cutoff']} > match_date {row['match_date']}"
        )


def test_backtest_has_required_columns():
    results = backtest(["WC2018"])
    required = [
        "match_date", "home_team", "away_team", "home_score", "away_score",
        "outcome", "p_home", "p_draw", "p_away", "train_cutoff", "tournament_id",
    ]
    for col in required:
        assert col in results.columns, f"Missing column: {col}"


def test_backtest_probabilities_valid():
    results = backtest(["WC2018"])
    for _, row in results.iterrows():
        total = row["p_home"] + row["p_draw"] + row["p_away"]
        assert abs(total - 1.0) < 1e-4, f"Probs don't sum to 1: {total}"
        assert row["p_home"] >= 0
        assert row["p_draw"] >= 0
        assert row["p_away"] >= 0
