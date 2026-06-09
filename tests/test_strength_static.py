import pandas as pd
from src.data_loader import load_results
from src.strength_static import train_model


def test_train_model_returns_model():
    df = load_results()
    cutoff = pd.Timestamp("2010-06-01")
    model = train_model(df, cutoff)
    assert model is not None
    assert hasattr(model, "predict_match")


def test_predict_match_probabilities():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_model(df, cutoff)
    probs = model.predict_match("Brazil", "Germany", neutral=True)
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(p >= 0 for p in probs)


def test_predict_home_advantage():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_model(df, cutoff)
    probs_home = model.predict_match("Brazil", "Bolivia", neutral=False)
    probs_neutral = model.predict_match("Brazil", "Bolivia", neutral=True)
    assert probs_home[0] > probs_neutral[0]


def test_strong_team_beats_weak():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_model(df, cutoff)
    probs = model.predict_match("Brazil", "Luxembourg", neutral=True)
    assert probs[0] > 0.5
