"""Score-driven (GAS) dynamic Poisson model.

Each team has time-varying (attack_t, defense_t) states.
After each match, states update proportional to the score (gradient)
of the Poisson log-likelihood -- principled version of Elo.

Observation model:
    goals_home ~ Poisson(exp(intercept + att_home - def_away + gamma * home))
    goals_away ~ Poisson(exp(intercept + att_away - def_home))

Score-driven updates:
    att_home += omega * tw * (goals_h - mu_h)
    def_home += omega * tw * (mu_a - goals_a)
    att_away += omega * tw * (goals_a - mu_a)
    def_away += omega * tw * (mu_h - goals_h)

omega scaled by tournament importance (like K-factor in Elo).
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


_MAX_GOALS = 10
_GOALS = np.arange(_MAX_GOALS)


def _poisson_1x2(home_lambda: float, away_lambda: float) -> tuple:
    home_lambda = min(max(home_lambda, 0.15), 7.0)
    away_lambda = min(max(away_lambda, 0.15), 7.0)
    h_pmf = poisson.pmf(_GOALS, home_lambda)
    a_pmf = poisson.pmf(_GOALS, away_lambda)
    grid = np.outer(h_pmf, a_pmf)
    p_home = np.tril(grid, k=-1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, k=1).sum()
    total = p_home + p_draw + p_away
    return (p_home / total, p_draw / total, p_away / total)


def _tournament_weight_str(tournament: str) -> float:
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return 3.0
    if "confederations" in t or "copa america" in t or "euro" in t or "african" in t or "asian" in t:
        return 2.5
    if "qualif" in t:
        return 2.0
    return 1.0


def _precompute_tournament_weights(tournaments: pd.Series) -> np.ndarray:
    return np.array([_tournament_weight_str(str(t)) for t in tournaments])


class GASModel:
    """Score-driven dynamic Poisson model for match prediction."""

    def __init__(self, attack: dict, defense: dict, intercept: float,
                 home_adv: float, omega: float):
        self.attack = attack
        self.defense = defense
        self.intercept = intercept
        self.home_adv = home_adv
        self.omega = omega

    def predict_match(self, home: str, away: str, neutral: bool = False) -> tuple:
        att_h = self.attack.get(home, 0.0)
        def_h = self.defense.get(home, 0.0)
        att_a = self.attack.get(away, 0.0)
        def_a = self.defense.get(away, 0.0)

        gamma = 0 if neutral else self.home_adv
        mu_h = math.exp(self.intercept + att_h - def_a + gamma)
        mu_a = math.exp(self.intercept + att_a - def_h)

        return _poisson_1x2(mu_h, mu_a)


def _prepare_arrays(matches: pd.DataFrame) -> dict:
    """Convert DataFrame to arrays + team index for fast iteration."""
    teams_set = set(matches["home_team"]) | set(matches["away_team"])
    teams = sorted(teams_set)
    t2i = {t: i for i, t in enumerate(teams)}

    home_idx = np.array([t2i[t] for t in matches["home_team"]])
    away_idx = np.array([t2i[t] for t in matches["away_team"]])
    h_goals = matches["home_score"].astype(int).values
    a_goals = matches["away_score"].astype(int).values

    neutral_vals = matches["neutral"].fillna("FALSE")
    if neutral_vals.dtype == bool:
        is_neutral = neutral_vals.values.astype(float)
    else:
        is_neutral = np.array([1.0 if str(v).upper() == "TRUE" else 0.0 for v in neutral_vals])

    tw = _precompute_tournament_weights(matches["tournament"])

    return {
        "teams": teams, "t2i": t2i, "n_teams": len(teams),
        "home_idx": home_idx, "away_idx": away_idx,
        "h_goals": h_goals, "a_goals": a_goals,
        "is_neutral": is_neutral, "tw": tw,
    }


def _run_gas_fast(
    arrays: dict,
    intercept: float,
    home_adv: float,
    omega: float,
    init_attack: np.ndarray | None = None,
    init_defense: np.ndarray | None = None,
    compute_ll: bool = False,
    ll_start_idx: int = 0,
) -> tuple:
    """Fast GAS filter using numpy arrays. Returns (attack, defense, log_lik)."""
    n_teams = arrays["n_teams"]
    home_idx = arrays["home_idx"]
    away_idx = arrays["away_idx"]
    h_goals = arrays["h_goals"]
    a_goals = arrays["a_goals"]
    is_neutral = arrays["is_neutral"]
    tw = arrays["tw"]
    n = len(home_idx)

    att = np.zeros(n_teams) if init_attack is None else init_attack.copy()
    dfn = np.zeros(n_teams) if init_defense is None else init_defense.copy()

    total_ll = 0.0
    shrink = 0.9999
    log = math.log
    exp = math.exp

    for i in range(n):
        hi = home_idx[i]
        ai = away_idx[i]
        hg = h_goals[i]
        ag = a_goals[i]

        gamma = 0.0 if is_neutral[i] > 0.5 else home_adv
        lin_h = intercept + att[hi] - dfn[ai] + gamma
        lin_a = intercept + att[ai] - dfn[hi]

        # Clamp linear predictor to avoid overflow
        lin_h = max(min(lin_h, 2.7), -3.0)  # exp range ~0.05 to ~15
        lin_a = max(min(lin_a, 2.7), -3.0)

        mu_h = exp(lin_h)
        mu_a = exp(lin_a)

        if compute_ll and i >= ll_start_idx:
            total_ll += hg * log(mu_h) - mu_h
            total_ll += ag * log(mu_a) - mu_a

        step = omega * tw[i]
        score_h = hg - mu_h
        score_a = ag - mu_a

        att[hi] = (att[hi] + step * score_h) * shrink
        dfn[hi] = (dfn[hi] - step * score_a) * shrink  # conceded less = better
        att[ai] = (att[ai] + step * score_a) * shrink
        dfn[ai] = (dfn[ai] - step * score_h) * shrink

    return att, dfn, total_ll


def _run_gas_filter(
    matches: pd.DataFrame,
    intercept: float,
    home_adv: float,
    omega: float,
    return_ll: bool = False,
) -> dict | float:
    """Run GAS filter (compatibility wrapper)."""
    arrays = _prepare_arrays(matches)
    att, dfn, ll = _run_gas_fast(arrays, intercept, home_adv, omega, compute_ll=return_ll)

    if return_ll:
        return ll

    teams = arrays["teams"]
    return {
        "attack": {teams[i]: att[i] for i in range(len(teams))},
        "defense": {teams[i]: dfn[i] for i in range(len(teams))},
    }


def train_gas_model(df: pd.DataFrame, cutoff: pd.Timestamp) -> GASModel:
    """Train GAS model: optimize hyperparameters on data before cutoff."""
    train = df[df["date"] < cutoff].sort_values("date")
    train = train.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    # Precompute arrays once
    arrays = _prepare_arrays(train)

    # Split: warm-up on early data, optimize LL on recent 4 years
    opt_start = cutoff - pd.Timedelta(days=4 * 365)
    ll_start_idx = int((train["date"] >= opt_start).idxmax()) if (train["date"] >= opt_start).any() else 0

    def neg_ll(params):
        intercept, home_adv, log_omega = params
        omega = math.exp(log_omega)

        _, _, total_ll = _run_gas_fast(
            arrays, intercept, home_adv, omega,
            compute_ll=True, ll_start_idx=ll_start_idx,
        )

        reg = 10.0 * log_omega ** 2
        return -(total_ll - reg)

    x0 = np.array([0.25, 0.25, math.log(0.01)])
    result = minimize(neg_ll, x0, method="Nelder-Mead",
                      options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-1})

    best_intercept, best_home_adv, best_log_omega = result.x
    best_omega = math.exp(best_log_omega)

    # Final run with optimized params
    att, dfn, _ = _run_gas_fast(arrays, best_intercept, best_home_adv, best_omega)

    teams = arrays["teams"]
    return GASModel(
        attack={teams[i]: att[i] for i in range(len(teams))},
        defense={teams[i]: dfn[i] for i in range(len(teams))},
        intercept=best_intercept,
        home_adv=best_home_adv,
        omega=best_omega,
    )
