import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


# Precompute Poisson PMF table for speed
_MAX_GOALS = 10
_GOALS = np.arange(_MAX_GOALS)
_POISSON_TABLE = {}  # cache


def _poisson_1x2(home_lambda: float, away_lambda: float) -> tuple:
    """Compute (p_home, p_draw, p_away) from Poisson parameters."""
    home_lambda = np.clip(home_lambda, 0.15, 7.0)
    away_lambda = np.clip(away_lambda, 0.15, 7.0)

    h_pmf = poisson.pmf(_GOALS, home_lambda)
    a_pmf = poisson.pmf(_GOALS, away_lambda)
    # Outer product gives P(H=h, A=a)
    grid = np.outer(h_pmf, a_pmf)

    p_home = np.tril(grid, k=-1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, k=1).sum()

    total = p_home + p_draw + p_away
    return (p_home / total, p_draw / total, p_away / total)


def _batch_poisson_1x2(home_lambdas: np.ndarray, away_lambdas: np.ndarray) -> np.ndarray:
    """Vectorized: compute (N, 3) probabilities from arrays of lambdas."""
    home_lambdas = np.clip(home_lambdas, 0.15, 7.0)
    away_lambdas = np.clip(away_lambdas, 0.15, 7.0)

    n = len(home_lambdas)
    goals = np.arange(_MAX_GOALS)

    # Shape: (N, MAX_GOALS)
    h_pmf = poisson.pmf(goals[None, :], home_lambdas[:, None])
    a_pmf = poisson.pmf(goals[None, :], away_lambdas[:, None])

    # Shape: (N, MAX_GOALS, MAX_GOALS) - outer product per match
    grid = h_pmf[:, :, None] * a_pmf[:, None, :]

    # Masks for outcomes
    h_idx, a_idx = np.meshgrid(goals, goals, indexing="ij")
    home_mask = h_idx > a_idx
    draw_mask = h_idx == a_idx
    away_mask = h_idx < a_idx

    p_home = (grid * home_mask[None, :, :]).sum(axis=(1, 2))
    p_draw = (grid * draw_mask[None, :, :]).sum(axis=(1, 2))
    p_away = (grid * away_mask[None, :, :]).sum(axis=(1, 2))

    total = p_home + p_draw + p_away
    probs = np.column_stack([p_home / total, p_draw / total, p_away / total])
    return probs


class EloModel:
    """Elo rating system with goal difference adjustment."""

    def __init__(self, ratings: dict[str, float], home_adv: float, scale: float):
        self.ratings = ratings
        self.home_adv = home_adv
        self.scale = scale

    def predict_match(self, home: str, away: str, neutral: bool = False) -> tuple:
        """Return (p_home_win, p_draw, p_away_win)."""
        r_home = self.ratings.get(home, 1500.0)
        r_away = self.ratings.get(away, 1500.0)

        diff = r_home - r_away + (self.home_adv if not neutral else 0)

        home_lambda = np.exp(0.25 + diff / self.scale)
        away_lambda = np.exp(0.25 - diff / self.scale)

        return _poisson_1x2(home_lambda, away_lambda)


def _k_factor(tournament: str) -> float:
    """K factor: how much a single match can move ratings.

    Key insight: WC qualifiers (K=30) are discounted vs continental
    tournaments (K=50) because volume inflates ratings. Brazil plays
    18 CONMEBOL qualifiers per cycle, winning ~14, each adding small
    delta that accumulates. Continental finals (Copa, Euro) are higher
    stakes and better signal.
    """
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if any(x in t for x in ("confederations", "copa america", "euro", "african", "asian")):
        return 50.0
    if "qualif" in t:
        return 30.0  # was 40 — reduced to prevent qualifier volume inflation
    if "nations league" in t:
        return 35.0
    return 20.0


def _goal_diff_multiplier(goal_diff: int) -> float:
    """World Football Elo goal difference multiplier."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11.0 + gd) / 8.0


def _expected_result(rating_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


def compute_elo_ratings(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    initial_rating: float = 1500.0,
    home_adv_elo: float = 100.0,
    decay_per_year: float = 0.94,
    quality_exponent: float = 0.3,
) -> dict[str, float]:
    """Compute Elo ratings with time decay and opponent quality adjustment.

    Two corrections vs standard Elo:
    1. Time decay: between matches, ratings regress toward the mean (1500)
       at rate decay_per_year. Half-life ~11 years at 0.94.
       This prevents Chile's 2015 Copa America from inflating their 2026 rating.
    2. Opponent quality: K is multiplied by (opponent_elo / 1500)^exponent.
       Beating Germany (1800) gives K*1.06, beating Bolivia (1200) gives K*0.95.
       This compounds with the expected result adjustment already in Elo.
    """
    train = df[df["date"] < cutoff].sort_values("date")
    train = train.dropna(subset=["home_score", "away_score"])

    ratings: dict[str, float] = {}
    last_match_date: dict[str, pd.Timestamp] = {}

    for _, row in train.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_score = int(row["home_score"])
        a_score = int(row["away_score"])
        tournament = str(row.get("tournament", "Friendly"))
        is_neutral = str(row.get("neutral", "FALSE")).upper() == "TRUE"
        match_date = row["date"]

        r_h = ratings.get(home, initial_rating)
        r_a = ratings.get(away, initial_rating)

        # --- Time decay: regress toward mean since last match ---
        for team, r in [(home, r_h), (away, r_a)]:
            if team in last_match_date:
                years_gap = (match_date - last_match_date[team]).days / 365.25
                if years_gap > 0:
                    decay = decay_per_year ** years_gap
                    new_r = initial_rating + (r - initial_rating) * decay
                    if team == home:
                        r_h = new_r
                    else:
                        r_a = new_r
                    ratings[team] = new_r

        ha = 0 if is_neutral else home_adv_elo
        diff = r_h + ha - r_a
        exp_h = _expected_result(diff)

        if h_score > a_score:
            actual_h = 1.0
        elif h_score == a_score:
            actual_h = 0.5
        else:
            actual_h = 0.0

        k = _k_factor(tournament)
        g = _goal_diff_multiplier(h_score - a_score)

        # --- Opponent quality: scale K by opponent strength ---
        # Home team's update scaled by away team's quality (and vice versa)
        opp_quality_h = (max(r_a, 800) / initial_rating) ** quality_exponent
        opp_quality_a = (max(r_h, 800) / initial_rating) ** quality_exponent

        delta_h = k * g * opp_quality_h * (actual_h - exp_h)
        delta_a = k * g * opp_quality_a * ((1 - actual_h) - (1 - exp_h))

        ratings[home] = r_h + delta_h
        ratings[away] = r_a + delta_a
        last_match_date[home] = match_date
        last_match_date[away] = match_date

    # --- Final decay to cutoff: penalize teams that haven't played recently ---
    for team in ratings:
        if team in last_match_date:
            years_gap = (cutoff - last_match_date[team]).days / 365.25
            if years_gap > 0.1:  # more than ~1 month
                decay = decay_per_year ** years_gap
                ratings[team] = initial_rating + (ratings[team] - initial_rating) * decay

    return ratings


def train_elo_model(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    home_adv_elo: float = 100.0,
    scale: float = 600.0,
) -> EloModel:
    """Train Elo model: compute ratings up to cutoff, calibrate scale on recent data."""
    ratings = compute_elo_ratings(df, cutoff, home_adv_elo=home_adv_elo)

    # Calibrate scale on last 2 years of data
    cal_start = cutoff - pd.Timedelta(days=730)
    cal_data = df[(df["date"] >= cal_start) & (df["date"] < cutoff)].copy()
    cal_data = cal_data.dropna(subset=["home_score", "away_score"])

    if len(cal_data) < 50:
        return EloModel(ratings, home_adv_elo, scale)

    # Precompute elo diffs and outcomes for calibration set
    diffs = np.array([
        ratings.get(row["home_team"], 1500.0) - ratings.get(row["away_team"], 1500.0)
        + (0 if str(row.get("neutral", "FALSE")).upper() == "TRUE" else home_adv_elo)
        for _, row in cal_data.iterrows()
    ])

    outcomes = np.array([
        0 if int(row["home_score"]) > int(row["away_score"])
        else (1 if int(row["home_score"]) == int(row["away_score"]) else 2)
        for _, row in cal_data.iterrows()
    ])

    eye3 = np.eye(3)

    def cal_rps(s):
        home_lambdas = np.exp(0.25 + diffs / s)
        away_lambdas = np.exp(0.25 - diffs / s)
        probs = _batch_poisson_1x2(home_lambdas, away_lambdas)

        cum_pred = np.cumsum(probs, axis=1)
        cum_real = np.cumsum(eye3[outcomes], axis=1)
        return float(0.5 * np.mean(np.sum((cum_pred - cum_real) ** 2, axis=1)))

    result = minimize_scalar(cal_rps, bounds=(300, 1500), method="bounded")
    return EloModel(ratings, home_adv_elo, result.x)
