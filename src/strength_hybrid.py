"""Multi-feature hybrid model for World Cup prediction.

Blends 10 strength signals into a unified team rating:
LAYER 1 (backtestable on historical matches):
  1. Elo ratings - base signal from match history
  2. Squad market value - transfer market signal
  3. League composite - where players play daily
  4. Momentum - win % last 10 matches
  5. Defensive strength - goals conceded last 10
  6. Defending champion - historical repeat penalty

LAYER 2 (tournament-specific, not backtestable):
  7. Market odds - bookmaker implied probabilities
  8. Squad age fitness - distance from optimal age
  9. Coach tenure fitness - years in charge
  10. Host advantage - home soil boost
  11. Population - talent pool depth

Weights: Ridge regression (Layer 1) + informed priors (Layer 2).
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson

SQUAD_VALUES_PATH = Path(__file__).parent.parent / "data" / "squad_values.csv"

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


def _batch_poisson_1x2(home_lambdas, away_lambdas):
    home_lambdas = np.clip(home_lambdas, 0.15, 7.0)
    away_lambdas = np.clip(away_lambdas, 0.15, 7.0)
    goals = np.arange(_MAX_GOALS)
    h_pmf = poisson.pmf(goals[None, :], home_lambdas[:, None])
    a_pmf = poisson.pmf(goals[None, :], away_lambdas[:, None])
    grid = h_pmf[:, :, None] * a_pmf[:, None, :]
    h_idx, a_idx = np.meshgrid(goals, goals, indexing="ij")
    p_h = (grid * (h_idx > a_idx)[None, :, :]).sum(axis=(1, 2))
    p_d = (grid * (h_idx == a_idx)[None, :, :]).sum(axis=(1, 2))
    p_a = (grid * (h_idx < a_idx)[None, :, :]).sum(axis=(1, 2))
    total = p_h + p_d + p_a
    return np.column_stack([p_h / total, p_d / total, p_a / total])


def load_squad_values(path: Path = SQUAD_VALUES_PATH) -> dict[str, float]:
    """Load squad values and return {team: log_value_normalized}."""
    df = pd.read_csv(path)
    df["log_value"] = np.log(df["value_m_gbp"])
    mean_lv = df["log_value"].mean()
    std_lv = df["log_value"].std()
    df["norm_value"] = (df["log_value"] - mean_lv) / std_lv
    return dict(zip(df["team"], df["norm_value"]))


class HybridModel:
    """Multi-feature strength model."""

    def __init__(self, blended_strength: dict[str, float],
                 home_adv: float, scale: float, weights: dict[str, float]):
        self.strength = blended_strength
        self.home_adv = home_adv
        self.scale = scale
        self.weights = weights

    def predict_match(self, home: str, away: str, neutral: bool = False) -> tuple:
        s_home = self.strength.get(home, 0.0)
        s_away = self.strength.get(away, 0.0)
        diff = s_home - s_away + (self.home_adv if not neutral else 0)
        home_lambda = math.exp(0.25 + diff / self.scale)
        away_lambda = math.exp(0.25 - diff / self.scale)
        return _poisson_1x2(home_lambda, away_lambda)


def _blend_all_features(feature_dicts: dict[str, dict], weights: dict) -> dict:
    """Blend all features into unified strength score."""
    all_teams = set()
    for fd in feature_dicts.values():
        all_teams |= set(fd.keys())
    blended = {}
    for team in all_teams:
        s = sum(weights.get(fname, 0.0) * fd.get(team, 0.0)
                for fname, fd in feature_dicts.items())
        blended[team] = s
    return blended


def _zscore_dict(d: dict) -> dict:
    vals = np.array(list(d.values()))
    m, s = vals.mean(), vals.std()
    if s < 1e-8: s = 1.0
    return {t: (v - m) / s for t, v in d.items()}


def train_hybrid_model(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    squad_values: dict[str, float] | None = None,
    use_market: bool = True,
    market_weight: float = 0.35,
) -> HybridModel:
    """Train multi-feature hybrid model with two-layer blending."""
    from src.strength_elo import compute_elo_ratings
    from src.features import (
        compute_momentum, compute_defensive_strength, compute_league_composite,
        compute_defending_champion_feature, compute_market_strength,
        compute_age_fitness, compute_coach_tenure,
        compute_host_advantage, compute_population_factor,
        compute_diversity_index, compute_composition_threshold,
        compute_frontrunner_curse,
    )

    ratings = compute_elo_ratings(df, cutoff)
    if squad_values is None:
        squad_values = load_squad_values()

    # === Compute ALL features ===
    league_composite = compute_league_composite()
    momentum_norm = _zscore_dict(compute_momentum(df, cutoff, n_matches=10, elo_ratings=ratings))
    defense_norm = _zscore_dict(compute_defensive_strength(df, cutoff, n_matches=10))

    all_teams = set(ratings.keys()) | set(squad_values.keys()) | set(momentum_norm.keys())
    defending = compute_defending_champion_feature(cutoff, all_teams)

    # Normalize Elo
    elo_vals = np.array(list(ratings.values()))
    elo_mean, elo_std = elo_vals.mean(), elo_vals.std()
    elo_norm = {t: (ratings.get(t, 1500.0) - elo_mean) / elo_std for t in all_teams}

    # Layer 2 features (tournament-specific)
    age_fitness = compute_age_fitness()
    coach_tenure = compute_coach_tenure()
    host_adv = compute_host_advantage(all_teams)
    population = compute_population_factor()
    diversity = compute_diversity_index()
    composition = compute_composition_threshold()
    frontrunner = compute_frontrunner_curse()

    # === LAYER 1: Calibrate on historical competitive matches ===
    cal_start = cutoff - pd.Timedelta(days=1460)
    cal_data = df[(df["date"] >= cal_start) & (df["date"] < cutoff)].copy()
    cal_data = cal_data.dropna(subset=["home_score", "away_score"])
    competitive_kw = ["FIFA World Cup", "UEFA Euro", "Copa Am", "Africa Cup",
                      "AFC Asian Cup", "CONCACAF Gold Cup", "Nations League",
                      "qualification", "Qualifying"]
    cal_data = cal_data[cal_data["tournament"].str.contains(
        "|".join(competitive_kw), case=False, na=False
    )]

    n = len(cal_data)
    home_teams = cal_data["home_team"].values
    away_teams = cal_data["away_team"].values

    # Backtestable features
    cal_feat_names = ["elo", "value", "league", "momentum", "defense", "champion"]
    cal_feat_dicts = [elo_norm, squad_values, league_composite,
                      momentum_norm, defense_norm, defending]

    feat_h = np.zeros((n, len(cal_feat_names)))
    feat_a = np.zeros((n, len(cal_feat_names)))
    for fi, fd in enumerate(cal_feat_dicts):
        for i in range(n):
            feat_h[i, fi] = fd.get(home_teams[i], 0.0)
            feat_a[i, fi] = fd.get(away_teams[i], 0.0)

    is_neutral = np.array([
        1.0 if str(r.get("neutral", "FALSE")).upper() == "TRUE" else 0.0
        for _, r in cal_data.iterrows()
    ])
    outcomes = np.array([
        0 if int(r["home_score"]) > int(r["away_score"])
        else (1 if int(r["home_score"]) == int(r["away_score"]) else 2)
        for _, r in cal_data.iterrows()
    ])
    eye3 = np.eye(3)

    # Ridge regression for feature importance
    from sklearn.linear_model import RidgeCV

    margin = cal_data["home_score"].astype(int).values - cal_data["away_score"].astype(int).values
    feat_diff = feat_h - feat_a
    feat_diff_with_ha = np.column_stack([feat_diff, 1 - is_neutral])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(feat_diff_with_ha, margin)
    raw_coefs = ridge.coef_[:len(cal_feat_names)]

    abs_coefs = np.abs(raw_coefs)
    raw_weights = abs_coefs / abs_coefs.sum() if abs_coefs.sum() > 0 else np.ones(len(cal_feat_names)) / len(cal_feat_names)

    # Informed priors for backtestable features
    PRIOR = {"elo": 0.45, "value": 0.15, "league": 0.10,
             "momentum": 0.10, "defense": 0.10, "champion": 0.10}
    MIN_FLOORS = {"elo": 0.35, "value": 0.05, "league": 0.05,
                  "momentum": 0.05, "defense": 0.05, "champion": 0.0}

    model_w = {}
    for i, name in enumerate(cal_feat_names):
        w = 0.6 * float(raw_weights[i]) + 0.4 * PRIOR[name]
        model_w[name] = max(w, MIN_FLOORS[name])

    champ_w = model_w.pop("champion")
    total_w = sum(model_w.values())
    for name in model_w:
        model_w[name] /= total_w
    model_w["champion"] = champ_w

    print(f"    Ridge coefs: {dict(zip(cal_feat_names, raw_coefs.round(4)))}")
    print(f"    L1 weights: {', '.join(f'{k}={v:.2f}' for k,v in model_w.items())}")

    # Calibrate scale + home_adv
    best_rps = float("inf")
    best_scale = 2.0
    best_ha = 0.15

    w_arr = np.array([model_w.get(n, 0.0) for n in cal_feat_names])
    s_h = feat_h @ w_arr
    s_a = feat_a @ w_arr

    for scale_test in np.arange(1.0, 4.5, 0.25):
        for ha_test in np.arange(0.0, 0.45, 0.05):
            diff = s_h - s_a + ha_test * (1 - is_neutral)
            h_lam = np.exp(0.25 + diff / scale_test)
            a_lam = np.exp(0.25 - diff / scale_test)

            probs = _batch_poisson_1x2(h_lam, a_lam)
            cum_p = np.cumsum(probs, axis=1)
            cum_r = np.cumsum(eye3[outcomes], axis=1)
            rps_val = 0.5 * np.mean(np.sum((cum_p - cum_r) ** 2, axis=1))

            if rps_val < best_rps:
                best_rps = rps_val
                best_scale = scale_test
                best_ha = ha_test

    print(f"    Scale={best_scale:.2f}, home_adv={best_ha:.2f}, cal_RPS={best_rps:.4f}")

    # Build Layer 1 model blend
    l1_features = {
        "elo": elo_norm, "value": squad_values, "league": league_composite,
        "momentum": momentum_norm, "defense": defense_norm, "champion": defending,
    }
    model_blend = _blend_all_features(l1_features, model_w)

    # === LAYER 2: Tournament-specific features + market ===
    # Normalize model blend to z-scores
    model_blend_z = _zscore_dict(model_blend)

    # Tournament-specific feature weights (informed priors, not calibratable)
    L2_WEIGHTS = {
        "model": 0.38,        # Layer 1 model blend
        "market": 0.32,       # bookmaker odds (strongest single signal)
        "age": 0.05,          # squad age fitness
        "coach": 0.04,        # coach tenure
        "host": 0.02,         # home soil advantage
        "population": 0.02,   # talent pool depth
        "defense_l2": 0.06,   # extra weight on defensive strength
        "diversity": 0.03,    # diaspora recruitment breadth * league quality
        "composition": 0.03,  # historical threshold: 6+ penalty
        "frontrunner": 0.05,  # frontrunner curse / sweet spot boost
    }

    if not use_market:
        # Redistribute market weight to model
        L2_WEIGHTS["model"] += L2_WEIGHTS["market"]
        L2_WEIGHTS["market"] = 0.0

    l2_features = {
        "model": model_blend_z,
        "age": age_fitness,
        "coach": coach_tenure,
        "host": host_adv,
        "population": population,
        "defense_l2": defense_norm,
        "diversity": diversity,
        "composition": composition,
        "frontrunner": frontrunner,
    }

    if use_market:
        market = compute_market_strength()
        l2_features["market"] = market

    final_blend = _blend_all_features(l2_features, L2_WEIGHTS)

    # Print feature breakdown for top teams
    all_wc_teams = set()
    for fd in l2_features.values():
        all_wc_teams |= set(fd.keys())

    print(f"\n    Layer 2 weights: {', '.join(f'{k}={v:.0%}' for k,v in L2_WEIGHTS.items())}")
    print(f"    Total features: {len(cal_feat_names)} backtestable + "
          f"{len(l2_features)-1} tournament-specific + market")

    final_weights = {**{f"L1_{k}": v for k, v in model_w.items()}, **L2_WEIGHTS}
    return HybridModel(final_blend, best_ha, best_scale, final_weights)
