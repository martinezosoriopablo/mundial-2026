import pandas as pd
from src.data_loader import load_results, filter_before


def test_load_results_columns():
    df = load_results()
    assert "date" in df.columns
    assert "home_team" in df.columns
    assert "away_team" in df.columns
    assert "home_score" in df.columns
    assert "away_score" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_results_not_empty():
    df = load_results()
    assert len(df) > 40000


def test_filter_before():
    df = load_results()
    cutoff = pd.Timestamp("2010-06-01")
    filtered = filter_before(df, cutoff)
    assert filtered["date"].max() < cutoff


def test_filter_before_preserves_data():
    df = load_results()
    cutoff = pd.Timestamp("2020-01-01")
    filtered = filter_before(df, cutoff)
    assert len(filtered) > 0
    assert len(filtered) < len(df)
