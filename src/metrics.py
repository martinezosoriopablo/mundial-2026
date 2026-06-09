import numpy as np


def rps(probs: np.ndarray, outcome_idx: int) -> float:
    """Ranked Probability Score for ordered 3-way outcome (H/D/A)."""
    cumulative_pred = np.cumsum(probs)
    cumulative_real = np.cumsum(np.eye(3)[outcome_idx])
    return float(0.5 * np.sum((cumulative_pred - cumulative_real) ** 2))


def log_loss_1x2(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Mean log-loss over match predictions. probs shape (N, 3), outcomes shape (N,)."""
    probs = np.clip(probs, eps, 1 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n = len(outcomes)
    return float(-np.sum(np.log(probs[np.arange(n), outcomes])) / n)


def brier_multi(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean multiclass Brier score. probs (N, 3), outcomes (N,)."""
    one_hot = np.eye(3)[outcomes]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error across all 3 classes."""
    total = 0.0
    n = len(outcomes)
    for cls in range(3):
        p = probs[:, cls]
        y = (outcomes == cls).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            avg_pred = p[mask].mean()
            avg_true = y[mask].mean()
            total += mask.sum() * abs(avg_pred - avg_true)
    return float(total / (n * 3))


def match_outcome(home_score: int, away_score: int) -> int:
    """Return 0=home win, 1=draw, 2=away win."""
    if home_score > away_score:
        return 0
    elif home_score == away_score:
        return 1
    else:
        return 2
