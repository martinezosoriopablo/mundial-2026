import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class StaticModel:
    def __init__(self, teams, attack, defense, home_adv, intercept):
        self.teams = teams
        self.team_to_idx = {t: i for i, t in enumerate(teams)}
        self.attack = attack
        self.defense = defense
        self.home_adv = home_adv
        self.intercept = intercept

    def predict_match(self, home: str, away: str, neutral: bool = False) -> tuple:
        """Return (p_home_win, p_draw, p_away_win)."""
        hi = self.team_to_idx.get(home)
        ai = self.team_to_idx.get(away)
        if hi is None or ai is None:
            return (1 / 3, 1 / 3, 1 / 3)

        home_lambda = np.exp(
            self.intercept
            + self.attack[hi]
            - self.defense[ai]
            + (self.home_adv if not neutral else 0)
        )
        away_lambda = np.exp(
            self.intercept + self.attack[ai] - self.defense[hi]
        )

        home_lambda = np.clip(home_lambda, 0.1, 8.0)
        away_lambda = np.clip(away_lambda, 0.1, 8.0)

        max_goals = 10
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for h in range(max_goals):
            for a in range(max_goals):
                p = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)
                if h > a:
                    p_home += p
                elif h == a:
                    p_draw += p
                else:
                    p_away += p

        total = p_home + p_draw + p_away
        return (p_home / total, p_draw / total, p_away / total)


def train_model(
    df: pd.DataFrame, cutoff: pd.Timestamp, half_life_years: float = 2.0
) -> StaticModel:
    """Train Poisson model on matches before cutoff with temporal decay."""
    train = df[df["date"] < cutoff].copy()
    train = train.dropna(subset=["home_score", "away_score"])
    train["home_score"] = train["home_score"].astype(int)
    train["away_score"] = train["away_score"].astype(int)

    # Temporal weights
    days_diff = (cutoff - train["date"]).dt.days.values
    half_life_days = half_life_years * 365.25
    weights = np.exp(-np.log(2) * days_diff / half_life_days)

    # Get all teams
    all_teams = pd.concat([train["home_team"], train["away_team"]]).unique()
    teams = sorted(all_teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Build arrays
    home_idx = train["home_team"].map(team_to_idx).values
    away_idx = train["away_team"].map(team_to_idx).values
    home_goals = train["home_score"].values.astype(float)
    away_goals = train["away_score"].values.astype(float)
    is_neutral = train["neutral"].fillna(False).values
    # Handle string "TRUE"/"FALSE" as well as bool
    if is_neutral.dtype == object:
        is_neutral = np.array([str(v).upper() == "TRUE" for v in is_neutral])
    is_neutral = is_neutral.astype(float)

    reg_lambda = 0.01

    def neg_log_lik(params):
        intercept = params[0]
        home_adv = params[1]
        attack = params[2 : 2 + n_teams]
        defense = params[2 + n_teams : 2 + 2 * n_teams]

        mu_h = np.exp(
            intercept
            + attack[home_idx]
            - defense[away_idx]
            + home_adv * (1 - is_neutral)
        )
        mu_a = np.exp(intercept + attack[away_idx] - defense[home_idx])

        mu_h = np.clip(mu_h, 1e-6, 20)
        mu_a = np.clip(mu_a, 1e-6, 20)

        ll_h = weights * (home_goals * np.log(mu_h) - mu_h)
        ll_a = weights * (away_goals * np.log(mu_a) - mu_a)

        reg = reg_lambda * (np.sum(attack**2) + np.sum(defense**2))

        return -(np.sum(ll_h) + np.sum(ll_a)) + reg

    x0 = np.zeros(2 + 2 * n_teams)
    x0[0] = 0.3

    result = minimize(
        neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-8}
    )

    params = result.x
    intercept = params[0]
    home_adv = params[1]
    attack = params[2 : 2 + n_teams]
    defense = params[2 + n_teams : 2 + 2 * n_teams]

    # Zero-sum constraint
    attack = attack - attack.mean()
    defense = defense - defense.mean()

    return StaticModel(teams, attack, defense, home_adv, intercept)
