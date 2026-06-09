import pandas as pd
from src.data_loader import load_results
from src.strength_gas import train_gas_model, _run_gas_filter


def test_gas_model_trains():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_gas_model(df, cutoff)
    assert model is not None
    assert hasattr(model, "predict_match")
    assert model.omega > 0


def test_gas_probabilities_valid():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_gas_model(df, cutoff)
    probs = model.predict_match("Brazil", "Germany", neutral=True)
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(p >= 0 for p in probs)


def test_gas_strong_beats_weak():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_gas_model(df, cutoff)
    probs = model.predict_match("Brazil", "Luxembourg", neutral=True)
    assert probs[0] > 0.5


def test_gas_home_advantage():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_gas_model(df, cutoff)
    probs_home = model.predict_match("Brazil", "Bolivia", neutral=False)
    probs_neutral = model.predict_match("Brazil", "Bolivia", neutral=True)
    assert probs_home[0] > probs_neutral[0]


def test_gas_states_evolve():
    """States should change over time — not static."""
    df = load_results()
    states_2010 = _run_gas_filter(
        df[df["date"] < "2010-06-01"].sort_values("date").dropna(subset=["home_score", "away_score"]),
        intercept=0.25, home_adv=0.25, omega=0.01,
    )
    states_2018 = _run_gas_filter(
        df[df["date"] < "2018-06-01"].sort_values("date").dropna(subset=["home_score", "away_score"]),
        intercept=0.25, home_adv=0.25, omega=0.01,
    )
    # Spain's attack should differ between 2010 and 2018
    att_spain_2010 = states_2010["attack"].get("Spain", 0)
    att_spain_2018 = states_2018["attack"].get("Spain", 0)
    assert att_spain_2010 != att_spain_2018
