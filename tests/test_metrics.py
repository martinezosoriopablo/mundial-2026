import numpy as np
from src.metrics import rps, log_loss_1x2, brier_multi, calibration_ece


def test_rps_perfect_prediction():
    score = rps(np.array([1.0, 0.0, 0.0]), 0)
    assert score == 0.0


def test_rps_worst_prediction():
    score = rps(np.array([1.0, 0.0, 0.0]), 2)
    assert score == 1.0


def test_rps_uniform():
    score = rps(np.array([1 / 3, 1 / 3, 1 / 3]), 0)
    expected = 0.5 * ((1 / 3 - 1) ** 2 + (2 / 3 - 1) ** 2)
    assert abs(score - expected) < 1e-10


def test_rps_symmetric():
    s1 = rps(np.array([0.0, 0.0, 1.0]), 0)
    s2 = rps(np.array([0.0, 1.0, 0.0]), 0)
    assert s1 > s2


def test_log_loss_perfect():
    probs = np.array([[0.99, 0.005, 0.005]])
    outcomes = np.array([0])
    score = log_loss_1x2(probs, outcomes)
    assert score < 0.02


def test_brier_perfect():
    probs = np.array([[1.0, 0.0, 0.0]])
    outcomes = np.array([0])
    score = brier_multi(probs, outcomes)
    assert score == 0.0


def test_calibration_ece_perfect():
    n = 1000
    probs = np.column_stack([
        np.full(n, 0.5),
        np.full(n, 0.3),
        np.full(n, 0.2),
    ])
    rng = np.random.default_rng(42)
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    ece = calibration_ece(probs, outcomes, n_bins=5)
    assert ece < 0.1
