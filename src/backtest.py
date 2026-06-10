"""Walk-forward backtesting harness for tournament prediction.

Honest out-of-sample validation over ~14 major tournaments (2010-2024).
Primary metric: RPS at match level (1-X-2 ordinal).
Secondary: log-loss, Brier multiclass, ECE calibration.

NO leakage: for each tournament T, model trains only on date < start(T).
"""

import math
import numpy as np
import pandas as pd
from collections import OrderedDict
from dataclasses import dataclass
from scipy.stats import poisson
from sklearn.linear_model import RidgeCV

from src.data_loader import load_results
from src.strength_elo import compute_elo_ratings
from src.features import (
    compute_defensive_strength, compute_league_composite,
)


# ============================================================================
# TOURNAMENT PANEL
# ============================================================================

TOURNAMENT_PANEL = OrderedDict([
    ("WC2010",  ("FIFA World Cup",  "2010-06-11", "2010-07-12")),
    ("CA2011",  ("Copa América",    "2011-07-01", "2011-07-25")),
    ("Euro2012",("UEFA Euro",       "2012-06-08", "2012-07-02")),
    ("WC2014",  ("FIFA World Cup",  "2014-06-12", "2014-07-14")),
    ("CA2015",  ("Copa América",    "2015-06-11", "2015-07-05")),
    ("CA2016",  ("Copa América",    "2016-06-03", "2016-06-27")),
    ("Euro2016",("UEFA Euro",       "2016-06-10", "2016-07-11")),
    ("WC2018",  ("FIFA World Cup",  "2018-06-14", "2018-07-16")),
    ("CA2019",  ("Copa América",    "2019-06-14", "2019-07-08")),
    ("Euro2020",("UEFA Euro",       "2021-06-11", "2021-07-12")),
    ("CA2021",  ("Copa América",    "2021-06-13", "2021-07-11")),
    ("WC2022",  ("FIFA World Cup",  "2022-11-20", "2022-12-19")),
    ("CA2024",  ("Copa América",    "2024-06-20", "2024-07-15")),
    ("Euro2024",("UEFA Euro",       "2024-06-14", "2024-07-15")),
])


# ============================================================================
# MODEL
# ============================================================================

_MAX_GOALS = 10
_GOALS = np.arange(_MAX_GOALS)


@dataclass
class Model:
    """Trained model: strength ratings + calibrated parameters."""
    strength: dict[str, float]
    home_adv: float
    scale: float
    weights: dict[str, float]
    rho: float = 0.0  # Dixon-Coles correlation parameter


def _zscore_dict(d: dict) -> dict:
    vals = np.array(list(d.values()))
    m, s = vals.mean(), vals.std()
    if s < 1e-8:
        s = 1.0
    return {t: (v - m) / s for t, v in d.items()}


def _dc_tau(h_goals, a_goals, lam_h, lam_a, rho):
    """Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts independent Poisson probabilities for (0,0), (1,0), (0,1), (1,1).
    rho < 0 means more draws than independent Poisson predicts.
    """
    if h_goals == 0 and a_goals == 0:
        return 1.0 - lam_h * lam_a * rho
    elif h_goals == 1 and a_goals == 0:
        return 1.0 + lam_a * rho
    elif h_goals == 0 and a_goals == 1:
        return 1.0 + lam_h * rho
    elif h_goals == 1 and a_goals == 1:
        return 1.0 - rho
    return 1.0


def _batch_poisson_1x2(home_lambdas, away_lambdas, rho=0.0):
    """Vectorized Poisson 1X2 with Dixon-Coles correction."""
    home_lambdas = np.clip(home_lambdas, 0.15, 7.0)
    away_lambdas = np.clip(away_lambdas, 0.15, 7.0)
    goals = np.arange(_MAX_GOALS)
    h_pmf = poisson.pmf(goals[None, :], home_lambdas[:, None])
    a_pmf = poisson.pmf(goals[None, :], away_lambdas[:, None])
    grid = h_pmf[:, :, None] * a_pmf[:, None, :]

    # Apply Dixon-Coles correction to low-scoring cells
    if rho != 0.0:
        # (0,0): multiply by 1 - lam_h * lam_a * rho
        grid[:, 0, 0] *= (1.0 - home_lambdas * away_lambdas * rho)
        # (1,0): multiply by 1 + lam_a * rho
        grid[:, 1, 0] *= (1.0 + away_lambdas * rho)
        # (0,1): multiply by 1 + lam_h * rho
        grid[:, 0, 1] *= (1.0 + home_lambdas * rho)
        # (1,1): multiply by 1 - rho
        grid[:, 1, 1] *= (1.0 - rho)
        # Clamp negatives (can happen with extreme rho)
        grid = np.maximum(grid, 0.0)

    h_idx, a_idx = np.meshgrid(goals, goals, indexing="ij")
    p_h = (grid * (h_idx > a_idx)[None, :, :]).sum(axis=(1, 2))
    p_d = (grid * (h_idx == a_idx)[None, :, :]).sum(axis=(1, 2))
    p_a = (grid * (h_idx < a_idx)[None, :, :]).sum(axis=(1, 2))
    total = p_h + p_d + p_a
    return np.column_stack([p_h / total, p_d / total, p_a / total])


def train_model(df: pd.DataFrame, cutoff: pd.Timestamp) -> Model:
    """Train Layer-1 model (no market data) on matches before cutoff.

    Elo + value + league + defense.
    Removed after ablation: champion (hurts RPS), momentum (hurts RPS).
    Ridge regression for weights, grid search for scale + home_adv.
    """
    ratings = compute_elo_ratings(df, cutoff)
    from src.strength_hybrid import load_squad_values
    squad_values = load_squad_values()
    league_composite = compute_league_composite()
    defense_norm = _zscore_dict(compute_defensive_strength(df, cutoff, 10))

    all_teams = set(ratings.keys()) | set(squad_values.keys())

    elo_vals = np.array(list(ratings.values()))
    elo_mean, elo_std = elo_vals.mean(), elo_vals.std()
    elo_norm = {t: (ratings.get(t, 1500.0) - elo_mean) / elo_std for t in all_teams}

    # Calibration window: 4 years of competitive matches before cutoff
    cal_start = cutoff - pd.Timedelta(days=1460)
    cal_data = df[(df["date"] >= cal_start) & (df["date"] < cutoff)].copy()
    cal_data = cal_data.dropna(subset=["home_score", "away_score"])
    comp_kw = ["FIFA World Cup", "UEFA Euro", "Copa Am", "Africa Cup",
               "AFC Asian Cup", "CONCACAF Gold Cup", "Nations League",
               "qualification", "Qualifying"]
    cal_data = cal_data[cal_data["tournament"].str.contains(
        "|".join(comp_kw), case=False, na=False)]

    n = len(cal_data)
    if n < 50:
        return Model(elo_norm, 0.2, 2.0, {"elo": 1.0})

    ht = cal_data["home_team"].values
    at = cal_data["away_team"].values

    feat_names = ["elo", "value", "league", "defense"]
    feat_dicts = [elo_norm, squad_values, league_composite, defense_norm]
    nf = len(feat_names)

    fh = np.zeros((n, nf))
    fa = np.zeros((n, nf))
    for fi, fd in enumerate(feat_dicts):
        for i in range(n):
            fh[i, fi] = fd.get(ht[i], 0.0)
            fa[i, fi] = fd.get(at[i], 0.0)

    is_neutral = np.array([
        1.0 if str(r.get("neutral", "FALSE")).upper() == "TRUE" else 0.0
        for _, r in cal_data.iterrows()])

    margin = cal_data["home_score"].astype(int).values - cal_data["away_score"].astype(int).values
    X = np.column_stack([fh - fa, 1 - is_neutral])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X, margin)
    raw_c = ridge.coef_[:nf]

    abs_c = np.abs(raw_c)
    rw = abs_c / abs_c.sum() if abs_c.sum() > 0 else np.ones(nf) / nf

    PRIOR = {"elo": 0.45, "value": 0.15, "league": 0.10, "defense": 0.10}
    FLOOR = {"elo": 0.35, "value": 0.05, "league": 0.05, "defense": 0.05}

    mw = {}
    for i, nm in enumerate(feat_names):
        mw[nm] = max(0.6 * float(rw[i]) + 0.4 * PRIOR[nm], FLOOR[nm])

    tot = sum(mw.values())
    for nm in mw:
        mw[nm] /= tot

    # Build blended strength
    l1 = dict(zip(feat_names, feat_dicts))

    blended = {}
    for team in all_teams:
        s = sum(mw.get(fname, 0.0) * fd.get(team, 0.0)
                for fname, fd in l1.items())
        blended[team] = s

    # Calibrate scale + home_adv + rho (Dixon-Coles) via RPS grid search
    outcomes = np.array([
        0 if int(r["home_score"]) > int(r["away_score"])
        else (1 if int(r["home_score"]) == int(r["away_score"]) else 2)
        for _, r in cal_data.iterrows()])
    eye3 = np.eye(3)

    w_arr = np.array([mw.get(nm, 0.0) for nm in feat_names])
    sh = fh @ w_arr
    sa = fa @ w_arr

    # First pass: find best scale + ha without DC (faster)
    best_rps, best_scale, best_ha = float("inf"), 2.0, 0.15
    for sc in np.arange(1.0, 4.5, 0.25):
        for ha in np.arange(0.0, 0.45, 0.05):
            d = sh - sa + ha * (1 - is_neutral)
            probs = _batch_poisson_1x2(np.exp(0.25 + d / sc), np.exp(0.25 - d / sc))
            cum_p = np.cumsum(probs, axis=1)
            cum_r = np.cumsum(eye3[outcomes], axis=1)
            rps_val = 0.5 * np.mean(np.sum((cum_p - cum_r) ** 2, axis=1))
            if rps_val < best_rps:
                best_rps, best_scale, best_ha = rps_val, sc, ha

    # Second pass: optimize rho with fixed scale + ha
    best_rho = 0.0
    d = sh - sa + best_ha * (1 - is_neutral)
    h_lam = np.exp(0.25 + d / best_scale)
    a_lam = np.exp(0.25 - d / best_scale)
    for rho_test in np.arange(-0.15, 0.05, 0.01):
        probs = _batch_poisson_1x2(h_lam, a_lam, rho=rho_test)
        cum_p = np.cumsum(probs, axis=1)
        cum_r = np.cumsum(eye3[outcomes], axis=1)
        rps_val = 0.5 * np.mean(np.sum((cum_p - cum_r) ** 2, axis=1))
        if rps_val < best_rps:
            best_rps, best_rho = rps_val, rho_test

    return Model(blended, best_ha, best_scale, mw, rho=best_rho)


def predict_match(model: Model, home: str, away: str,
                  neutral: bool = True) -> tuple[float, float, float]:
    """Predict (pHome, pDraw, pAway) with Dixon-Coles correction."""
    s_h = model.strength.get(home, 0.0)
    s_a = model.strength.get(away, 0.0)
    ha = model.home_adv if not neutral else 0.0
    diff = s_h - s_a + ha
    lam_h = min(max(math.exp(0.25 + diff / model.scale), 0.15), 7.0)
    lam_a = min(max(math.exp(0.25 - diff / model.scale), 0.15), 7.0)

    h_pmf = poisson.pmf(_GOALS, lam_h)
    a_pmf = poisson.pmf(_GOALS, lam_a)
    grid = np.outer(h_pmf, a_pmf)

    # Dixon-Coles correction for low-scoring outcomes
    rho = model.rho
    if rho != 0.0:
        grid[0, 0] *= _dc_tau(0, 0, lam_h, lam_a, rho)
        grid[1, 0] *= _dc_tau(1, 0, lam_h, lam_a, rho)
        grid[0, 1] *= _dc_tau(0, 1, lam_h, lam_a, rho)
        grid[1, 1] *= _dc_tau(1, 1, lam_h, lam_a, rho)
        grid = np.maximum(grid, 0.0)

    p_h = np.tril(grid, k=-1).sum()
    p_d = np.trace(grid)
    p_a = np.triu(grid, k=1).sum()
    total = p_h + p_d + p_a
    return (p_h / total, p_d / total, p_a / total)


# ============================================================================
# METRICS
# ============================================================================

def rps(probs: np.ndarray, outcome_idx: int) -> float:
    """Ranked Probability Score for ordinal 1-X-2. Lower is better."""
    cum_p = np.cumsum(probs)
    actual = np.zeros(3)
    actual[outcome_idx] = 1.0
    cum_a = np.cumsum(actual)
    return 0.5 * np.sum((cum_p - cum_a) ** 2)


def logloss(probs: np.ndarray, outcome_idx: int, eps: float = 1e-15) -> float:
    """Log-loss for multiclass 1X2."""
    p = np.clip(probs[outcome_idx], eps, 1.0 - eps)
    return -math.log(p)


def brier_multiclass(probs: np.ndarray, outcome_idx: int) -> float:
    """Brier score for multiclass 1X2."""
    actual = np.zeros(3)
    actual[outcome_idx] = 1.0
    return np.sum((probs - actual) ** 2)


# ============================================================================
# BACKTEST
# ============================================================================

def _get_tournament_matches(df: pd.DataFrame, pattern: str,
                            start: str, end: str) -> pd.DataFrame:
    """Extract matches for a specific tournament edition."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    mask = (
        df["tournament"].str.contains(pattern, case=False, na=False)
        & ~df["tournament"].str.contains("qualif", case=False, na=False)
        & (df["date"] >= start_dt)
        & (df["date"] <= end_dt)
    )
    return df[mask].copy()


def backtest(tournaments: list[str] | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Run walk-forward backtest over tournament panel.

    For each tournament T:
      1. Train model on all matches with date < start(T)
      2. Predict every match in T
      3. Score predictions (RPS, log-loss, Brier)

    Returns DataFrame with one row per match.
    """
    df = load_results()
    df["date"] = pd.to_datetime(df["date"])

    if tournaments is None:
        tournaments = list(TOURNAMENT_PANEL.keys())

    results = []

    for tkey in tournaments:
        if tkey not in TOURNAMENT_PANEL:
            print(f"  WARNING: {tkey} not in panel, skipping")
            continue

        pattern, start, end = TOURNAMENT_PANEL[tkey]
        cutoff = pd.Timestamp(start)

        t_matches = _get_tournament_matches(df, pattern, start, end)
        if len(t_matches) == 0:
            if verbose:
                print(f"  {tkey}: no matches found, skipping")
            continue

        # Train on data STRICTLY before tournament start
        train_data = df[df["date"] < cutoff].copy()
        assert train_data["date"].max() < cutoff, \
            f"LEAK: train max date {train_data['date'].max()} >= cutoff {cutoff}"

        if verbose:
            print(f"  {tkey}: train {len(train_data):,} matches "
                  f"(to {train_data['date'].max().date()}), "
                  f"test {len(t_matches)} matches")

        model = train_model(train_data, cutoff)

        for _, match in t_matches.iterrows():
            home = match["home_team"]
            away = match["away_team"]
            hs = int(match["home_score"])
            as_ = int(match["away_score"])
            is_neutral = str(match.get("neutral", "FALSE")).upper() == "TRUE"

            if hs > as_:
                outcome = 0
            elif hs == as_:
                outcome = 1
            else:
                outcome = 2

            p_h, p_d, p_a = predict_match(model, home, away, neutral=is_neutral)
            probs = np.array([p_h, p_d, p_a])

            results.append({
                "tournament": tkey,
                "date": match["date"],
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": as_,
                "outcome": outcome,
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "rps": rps(probs, outcome),
                "logloss": logloss(probs, outcome),
                "brier": brier_multiclass(probs, outcome),
                "neutral": is_neutral,
            })

    return pd.DataFrame(results)


def calibration_report(bt: pd.DataFrame, n_bins: int = 10) -> dict:
    """ECE + reliability bins for H/D/A."""
    report = {}
    outcome_names = ["home", "draw", "away"]
    prob_cols = ["p_home", "p_draw", "p_away"]

    total_ece = 0.0
    total_count = 0

    for oi, (oname, pcol) in enumerate(zip(outcome_names, prob_cols)):
        probs = bt[pcol].values
        actuals = (bt["outcome"].values == oi).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_data = []
        ece_weighted = 0.0

        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            mask = (probs >= lo) & (probs < hi)
            if b == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)

            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue

            avg_pred = probs[mask].mean()
            avg_actual = actuals[mask].mean()
            gap = abs(avg_pred - avg_actual)
            ece_weighted += gap * n_in_bin

            bin_data.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(n_in_bin),
                "avg_predicted": round(avg_pred, 4),
                "avg_actual": round(avg_actual, 4),
                "gap": round(gap, 4),
            })

        ece = ece_weighted / len(probs) if len(probs) > 0 else 0.0
        total_ece += ece_weighted
        total_count += len(probs)

        report[oname] = {"ece": round(ece, 5), "bins": bin_data}

    report["overall_ece"] = round(total_ece / total_count, 5) if total_count > 0 else 0.0
    return report


def summary_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per tournament."""
    grouped = bt.groupby("tournament").agg(
        n_matches=("rps", "count"),
        rps_mean=("rps", "mean"),
        logloss_mean=("logloss", "mean"),
        brier_mean=("brier", "mean"),
    ).round(5)

    overall = pd.DataFrame([{
        "tournament": "OVERALL",
        "n_matches": len(bt),
        "rps_mean": round(bt["rps"].mean(), 5),
        "logloss_mean": round(bt["logloss"].mean(), 5),
        "brier_mean": round(bt["brier"].mean(), 5),
    }]).set_index("tournament")

    return pd.concat([grouped, overall])


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("  BACKTEST WALK-FORWARD: ~14 TOURNAMENTS (2010-2024)")
    print("  Primary metric: RPS match-level 1-X-2 (lower = better)")
    print("=" * 80)
    print()

    bt = backtest(verbose=True)

    print(f"\n{'=' * 80}")
    print("  RESULTS PER TOURNAMENT")
    print(f"{'=' * 80}")

    summary = summary_table(bt)
    print(f"\n  {'Tournament':<12} {'N':>5} {'RPS':>8} {'LogLoss':>9} {'Brier':>8}")
    print(f"  {'-' * 45}")
    for tkey, row in summary.iterrows():
        marker = " ***" if tkey == "OVERALL" else ""
        print(f"  {tkey:<12} {int(row['n_matches']):>5} "
              f"{row['rps_mean']:>8.5f} {row['logloss_mean']:>9.5f} "
              f"{row['brier_mean']:>8.5f}{marker}")

    rps_baseline = bt["rps"].mean()
    print(f"\n  RPS_baseline = {rps_baseline:.6f}")
    print(f"  (This is the number to beat. No change accepted unless it lowers this.)")

    # Calibration
    print(f"\n{'=' * 80}")
    print("  CALIBRATION REPORT")
    print(f"{'=' * 80}")

    cal = calibration_report(bt)
    print(f"\n  Overall ECE: {cal['overall_ece']:.5f}")
    for oname in ["home", "draw", "away"]:
        print(f"\n  {oname.upper()} (ECE={cal[oname]['ece']:.5f}):")
        print(f"    {'Bin':<10} {'N':>5} {'Predicted':>10} {'Actual':>10} {'Gap':>8}")
        print(f"    {'-' * 45}")
        for b in cal[oname]["bins"]:
            print(f"    {b['bin']:<10} {b['n']:>5} "
                  f"{b['avg_predicted']:>10.4f} {b['avg_actual']:>10.4f} "
                  f"{b['gap']:>8.4f}")

    # Save baseline
    with open("backtest_baseline.txt", "w") as f:
        f.write(f"RPS_baseline={rps_baseline:.6f}\n")
        f.write(f"n_matches={len(bt)}\n")
        f.write(f"n_tournaments={bt['tournament'].nunique()}\n")
        f.write(f"logloss={bt['logloss'].mean():.6f}\n")
        f.write(f"brier={bt['brier'].mean():.6f}\n")
        f.write(f"ece={cal['overall_ece']:.6f}\n")

    print(f"\n  Baseline saved to backtest_baseline.txt")
    return bt, cal


if __name__ == "__main__":
    main()
