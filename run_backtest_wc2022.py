"""Backtest fuera de muestra: WC 2022.

Entrena el modelo con datos SOLO hasta nov 2022, simula el mundial 2022
con Monte Carlo, y compara predicciones vs resultados reales.

Esto es validacion out-of-sample pura.
"""

import math
import numpy as np
import pandas as pd
from collections import Counter
from src.data_loader import load_results
from src.strength_elo import compute_elo_ratings
from src.strength_hybrid import load_squad_values, _zscore_dict, _blend_all_features, HybridModel
from src.features import (
    compute_momentum, compute_defensive_strength, compute_league_composite,
    compute_defending_champion_feature,
)
from run_wc2026 import sample_scoreline, sample_knockout_scoreline

# WC 2022 Groups (actual draw)
GROUPS_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

# Actual WC 2022 results
ACTUAL_RESULTS = {
    "champion": "Argentina",
    "runner_up": "France",
    "third": "Croatia",
    "fourth": "Morocco",
    "semifinalists": ["Argentina", "France", "Croatia", "Morocco"],
    "quarterfinalists": ["Argentina", "France", "Croatia", "Morocco",
                         "Netherlands", "Brazil", "England", "Portugal"],
    "group_winners": {
        "A": "Netherlands", "B": "England", "C": "Argentina", "D": "France",
        "E": "Japan", "F": "Morocco", "G": "Brazil", "H": "Portugal",
    },
    "group_runners": {
        "A": "Senegal", "B": "United States", "C": "Poland", "D": "Australia",
        "E": "Spain", "F": "Croatia", "G": "Switzerland", "H": "South Korea",
    },
    "group_eliminated": [
        "Qatar", "Ecuador", "Iran", "Wales", "Saudi Arabia", "Mexico",
        "Denmark", "Tunisia", "Costa Rica", "Germany", "Belgium", "Canada",
        "Serbia", "Cameroon", "Ghana", "Uruguay",
    ],
    # Key upsets
    "upsets": [
        "Saudi Arabia beat Argentina (groups)",
        "Japan beat Germany (groups)",
        "Japan beat Spain (groups)",
        "Morocco beat Belgium (groups)",
        "Australia beat Denmark (groups)",
        "South Korea beat Portugal (groups)",
        "Morocco beat Spain (R16)",
        "Morocco beat Portugal (QF)",
    ],
}

# WC 2022 pre-tournament betting favorites (approximate)
MARKET_2022 = {
    "Brazil": 19.0, "France": 12.5, "Argentina": 11.0, "England": 10.0,
    "Spain": 9.0, "Germany": 8.0, "Netherlands": 5.0, "Portugal": 5.0,
    "Belgium": 4.5, "Denmark": 3.0, "Croatia": 2.0, "Uruguay": 2.0,
    "Switzerland": 1.0, "Senegal": 1.0, "Mexico": 0.8, "United States": 0.5,
    "Poland": 0.5, "Japan": 0.3, "Morocco": 0.3, "South Korea": 0.2,
    "Serbia": 0.3, "Ecuador": 0.2, "Cameroon": 0.2, "Ghana": 0.1,
    "Wales": 0.2, "Tunisia": 0.1, "Iran": 0.1, "Australia": 0.1,
    "Canada": 0.1, "Costa Rica": 0.05, "Qatar": 0.1, "Saudi Arabia": 0.05,
}


def train_backtest_model(df, cutoff):
    """Train Layer 1 model (backtestable features only) for WC 2022."""
    from sklearn.linear_model import RidgeCV

    ratings = compute_elo_ratings(df, cutoff)
    squad_values = load_squad_values()  # approximate, not period-adjusted
    league_composite = compute_league_composite()
    momentum_norm = _zscore_dict(compute_momentum(df, cutoff, n_matches=10, elo_ratings=ratings))
    defense_norm = _zscore_dict(compute_defensive_strength(df, cutoff, n_matches=10))

    all_teams = set(ratings.keys()) | set(squad_values.keys()) | set(momentum_norm.keys())
    defending = compute_defending_champion_feature(cutoff, all_teams)

    elo_vals = np.array(list(ratings.values()))
    elo_mean, elo_std = elo_vals.mean(), elo_vals.std()
    elo_norm = {t: (ratings.get(t, 1500.0) - elo_mean) / elo_std for t in all_teams}

    # Calibrate on competitive matches (same as main model)
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

    margin = cal_data["home_score"].astype(int).values - cal_data["away_score"].astype(int).values
    feat_diff = feat_h - feat_a
    feat_diff_with_ha = np.column_stack([feat_diff, 1 - is_neutral])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(feat_diff_with_ha, margin)
    raw_coefs = ridge.coef_[:len(cal_feat_names)]

    abs_coefs = np.abs(raw_coefs)
    raw_weights = abs_coefs / abs_coefs.sum() if abs_coefs.sum() > 0 else np.ones(len(cal_feat_names)) / len(cal_feat_names)

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
    print(f"    L1 weights: {', '.join(f'{k}={v:.2f}' for k, v in model_w.items())}")

    # Calibrate scale + home_adv
    from scipy.stats import poisson
    _MAX_GOALS = 10
    _GOALS = np.arange(_MAX_GOALS)

    outcomes = np.array([
        0 if int(r["home_score"]) > int(r["away_score"])
        else (1 if int(r["home_score"]) == int(r["away_score"]) else 2)
        for _, r in cal_data.iterrows()
    ])
    eye3 = np.eye(3)

    w_arr = np.array([model_w.get(n, 0.0) for n in cal_feat_names])
    s_h = feat_h @ w_arr
    s_a = feat_a @ w_arr

    from src.strength_hybrid import _batch_poisson_1x2

    best_rps = float("inf")
    best_scale = 2.0
    best_ha = 0.15

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

    # Build final blend
    l1_features = {
        "elo": elo_norm, "value": squad_values, "league": league_composite,
        "momentum": momentum_norm, "defense": defense_norm, "champion": defending,
    }
    model_blend = _blend_all_features(l1_features, model_w)

    return HybridModel(model_blend, best_ha, best_scale, model_w)


def predict_lambdas(model, home, away, neutral=True):
    s_home = model.strength.get(home, 0.0)
    s_away = model.strength.get(away, 0.0)
    diff = s_home - s_away + (model.home_adv if not neutral else 0)
    home_lambda = math.exp(0.25 + diff / model.scale)
    away_lambda = math.exp(0.25 - diff / model.scale)
    return max(0.2, min(home_lambda, 6.0)), max(0.2, min(away_lambda, 6.0))


def sim_group_stage(model, groups, rng):
    standings = {}
    for gname, teams in groups.items():
        pts = [0] * 4
        gf = [0] * 4
        ga = [0] * 4
        for i in range(4):
            for j in range(i + 1, 4):
                lh, la = predict_lambdas(model, teams[i], teams[j], neutral=True)
                gh, ga_ = sample_scoreline(rng, lh, la)
                gf[i] += gh; ga[i] += ga_
                gf[j] += ga_; ga[j] += gh
                if gh > ga_: pts[i] += 3
                elif gh == ga_: pts[i] += 1; pts[j] += 1
                else: pts[j] += 3
        indices = list(range(4))
        indices.sort(key=lambda k: (pts[k], gf[k] - ga[k], gf[k], rng.random()), reverse=True)
        standings[gname] = [teams[k] for k in indices]
    return standings


def sim_ko_match(model, h, a, rng):
    lh, la = predict_lambdas(model, h, a, neutral=True)
    gh, ga, _, _ = sample_knockout_scoreline(rng, lh, la)
    return h if gh > ga else a


def sim_tournament_2022(model, rng):
    """Simulate WC 2022 bracket (32 teams, 8 groups of 4, top 2 advance)."""
    standings = sim_group_stage(model, GROUPS_2022, rng)

    winners = {g: standings[g][0] for g in standings}
    runners = {g: standings[g][1] for g in standings}

    # WC 2022 actual R16 bracket
    r16 = [
        (winners["A"], runners["B"]),
        (winners["C"], runners["D"]),
        (winners["B"], runners["A"]),
        (winners["D"], runners["C"]),
        (winners["E"], runners["F"]),
        (winners["G"], runners["H"]),
        (winners["F"], runners["E"]),
        (winners["H"], runners["G"]),
    ]

    r16w = [sim_ko_match(model, h, a, rng) for h, a in r16]

    qf = [(r16w[i], r16w[i + 1]) for i in range(0, 8, 2)]
    qfw = [sim_ko_match(model, h, a, rng) for h, a in qf]

    sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
    sfw = [sim_ko_match(model, h, a, rng) for h, a in sf]

    champ = sim_ko_match(model, sfw[0], sfw[1], rng)
    finalist = sfw[1] if champ == sfw[0] else sfw[0]

    sf_losers = []
    for i, (h, a) in enumerate(sf):
        sf_losers.append(a if sfw[i] == h else h)

    return champ, finalist, sf_losers, qfw, standings


def run_backtest(model, n_sims=10000, seed=42):
    rng = np.random.default_rng(seed)

    champion_count = Counter()
    finalist_count = Counter()
    sf_count = Counter()
    qf_count = Counter()
    group_winner_count = {g: Counter() for g in GROUPS_2022}
    group_runner_count = {g: Counter() for g in GROUPS_2022}
    group_pos_count = {g: {t: Counter() for t in teams}
                       for g, teams in GROUPS_2022.items()}

    print(f"\n  Simulando {n_sims:,} WC 2022...", flush=True)
    for sim in range(n_sims):
        if (sim + 1) % 2500 == 0:
            print(f"    {sim + 1:,}/{n_sims:,}...", flush=True)

        champ, finalist, sf_losers, qfw, standings = sim_tournament_2022(model, rng)

        champion_count[champ] += 1
        finalist_count[champ] += 1
        finalist_count[finalist] += 1
        sf_count[champ] += 1
        sf_count[finalist] += 1
        for t in sf_losers:
            sf_count[t] += 1
        for t in qfw:
            qf_count[t] += 1

        for g in standings:
            group_winner_count[g][standings[g][0]] += 1
            group_runner_count[g][standings[g][1]] += 1
            for pos, t in enumerate(standings[g]):
                group_pos_count[g][t][pos + 1] += 1

    return (champion_count, finalist_count, sf_count, qf_count,
            group_winner_count, group_runner_count, group_pos_count, n_sims)


def print_backtest_results(champion_count, finalist_count, sf_count, qf_count,
                           group_winner_count, group_runner_count,
                           group_pos_count, n_sims, model):
    N = n_sims

    # =====================================================================
    # 1. CHAMPION PREDICTIONS vs REALITY
    # =====================================================================
    print(f"\n{'=' * 85}")
    print(f"  BACKTEST WC 2022 - MODELO vs REALIDAD ({N:,} simulaciones)")
    print(f"{'=' * 85}")

    print(f"\n  {'#':<4} {'Equipo':<18} {'Modelo':>8} {'Mercado':>8}"
          f" {'Real':>12}")
    print(f"  {'-' * 70}")

    model_probs = {t: c / N * 100 for t, c in champion_count.items()}
    all_teams = sorted(
        set(list(model_probs.keys()) + list(MARKET_2022.keys())),
        key=lambda t: -(model_probs.get(t, 0) + MARKET_2022.get(t, 0)) / 2
    )

    for rank, team in enumerate(all_teams[:20], 1):
        p_mod = model_probs.get(team, 0)
        p_mkt = MARKET_2022.get(team, 0)
        actual = ""
        if team == ACTUAL_RESULTS["champion"]:
            actual = "CAMPEON"
        elif team == ACTUAL_RESULTS["runner_up"]:
            actual = "FINAL"
        elif team == ACTUAL_RESULTS["third"]:
            actual = "3ro"
        elif team == ACTUAL_RESULTS["fourth"]:
            actual = "4to (SF)"
        elif team in ACTUAL_RESULTS["quarterfinalists"]:
            actual = "QF"
        elif team in ACTUAL_RESULTS["group_eliminated"]:
            actual = "Eliminado"

        print(f"  {rank:<4} {team:<18} {p_mod:>7.1f}% {p_mkt:>7.1f}%"
              f" {actual:>12}")

    # =====================================================================
    # 2. DID THE MODEL CAPTURE THE CHAMPION?
    # =====================================================================
    champ = ACTUAL_RESULTS["champion"]
    p_champ = model_probs.get(champ, 0)
    rank_champ = sorted(model_probs.keys(), key=lambda t: -model_probs[t]).index(champ) + 1

    print(f"\n  {'=' * 60}")
    print(f"  VEREDICTO: El modelo le daba a {champ} (campeon real):")
    print(f"    Probabilidad: {p_champ:.1f}%")
    print(f"    Ranking: #{rank_champ}")
    print(f"    Mercado le daba: {MARKET_2022.get(champ, 0):.1f}%")

    # How likely was the actual top 4?
    top4_actual = ACTUAL_RESULTS["semifinalists"]
    p_top4 = [model_probs.get(t, 0) for t in top4_actual]
    print(f"\n  Top 4 real: {', '.join(top4_actual)}")
    print(f"  Probabilidad del modelo para cada uno:")
    for t, p in zip(top4_actual, p_top4):
        sf_p = sf_count.get(t, 0) / N * 100
        print(f"    {t:<18} Campeon={p:.1f}%  Semi={sf_p:.1f}%")

    # =====================================================================
    # 3. GROUP STAGE ACCURACY
    # =====================================================================
    print(f"\n\n{'=' * 85}")
    print(f"  FASE DE GRUPOS: PREDICCION vs REALIDAD")
    print(f"{'=' * 85}")

    correct_winners = 0
    correct_qualifiers = 0
    total_groups = 8

    for g in sorted(GROUPS_2022.keys()):
        teams = GROUPS_2022[g]
        actual_w = ACTUAL_RESULTS["group_winners"][g]
        actual_r = ACTUAL_RESULTS["group_runners"][g]

        print(f"\n  Grupo {g}  (Real: 1ro={actual_w}, 2do={actual_r})")
        print(f"  {'Equipo':<18} {'1ro':>7} {'2do':>7} {'3ro':>7} {'4to':>7}"
              f"  {'Real'}")
        print(f"  {'-' * 62}")

        team_data = []
        for t in teams:
            c = group_pos_count[g][t]
            p1 = c.get(1, 0) / N * 100
            p2 = c.get(2, 0) / N * 100
            p3 = c.get(3, 0) / N * 100
            p4 = c.get(4, 0) / N * 100
            team_data.append((t, p1, p2, p3, p4))

        team_data.sort(key=lambda x: -(x[1] + x[2]))
        for t, p1, p2, p3, p4 in team_data:
            actual_pos = ""
            if t == actual_w:
                actual_pos = "<- 1ro"
            elif t == actual_r:
                actual_pos = "<- 2do"
            elif t in ACTUAL_RESULTS["group_eliminated"]:
                actual_pos = "   elim"

            print(f"  {t:<18} {p1:>6.1f}% {p2:>6.1f}% {p3:>6.1f}% {p4:>6.1f}%"
                  f"  {actual_pos}")

        # Check accuracy
        pred_winner = max(teams, key=lambda t: group_pos_count[g][t].get(1, 0))
        if pred_winner == actual_w:
            correct_winners += 1
        pred_top2 = sorted(teams, key=lambda t: -(group_pos_count[g][t].get(1, 0) + group_pos_count[g][t].get(2, 0)))[:2]
        if set(pred_top2) == {actual_w, actual_r}:
            correct_qualifiers += 1

    print(f"\n  Precision grupos:")
    print(f"    Ganador de grupo correcto:    {correct_winners}/{total_groups}"
          f" ({correct_winners / total_groups * 100:.0f}%)")
    print(f"    Top 2 clasificados correcto:  {correct_qualifiers}/{total_groups}"
          f" ({correct_qualifiers / total_groups * 100:.0f}%)")

    # =====================================================================
    # 4. SURPRISE DETECTION
    # =====================================================================
    print(f"\n\n{'=' * 85}")
    print(f"  SORPRESAS DEL MUNDIAL 2022: LAS PREDIJO EL MODELO?")
    print(f"{'=' * 85}")

    surprises = [
        ("Japan 1ro grupo E (sobre Spain, Germany)", "Japan",
         group_pos_count["E"]["Japan"].get(1, 0) / N * 100),
        ("Morocco 1ro grupo F (sobre Belgium, Croatia)", "Morocco",
         group_pos_count["F"]["Morocco"].get(1, 0) / N * 100),
        ("Australia 2do grupo D (sobre Denmark)", "Australia",
         group_pos_count["D"]["Australia"].get(2, 0) / N * 100),
        ("South Korea 2do grupo H (sobre Uruguay)", "South Korea",
         group_pos_count["H"]["South Korea"].get(2, 0) / N * 100),
        ("Germany eliminada en grupos", "Germany",
         (group_pos_count["E"]["Germany"].get(3, 0) +
          group_pos_count["E"]["Germany"].get(4, 0)) / N * 100),
        ("Belgium eliminada en grupos", "Belgium",
         (group_pos_count["F"]["Belgium"].get(3, 0) +
          group_pos_count["F"]["Belgium"].get(4, 0)) / N * 100),
        ("Morocco semifinalista", "Morocco",
         sf_count.get("Morocco", 0) / N * 100),
        ("Argentina campeon", "Argentina",
         champion_count.get("Argentina", 0) / N * 100),
    ]

    print(f"\n  {'Sorpresa':<55} {'P(modelo)':>10}")
    print(f"  {'-' * 68}")
    for desc, team, prob in surprises:
        label = "Capturo!" if prob > 10 else ("Parcial" if prob > 3 else "NO")
        print(f"  {desc:<55} {prob:>9.1f}%  {label}")

    # =====================================================================
    # 5. BRIER SCORE / CALIBRATION
    # =====================================================================
    print(f"\n\n{'=' * 85}")
    print(f"  CALIBRACION DEL MODELO")
    print(f"{'=' * 85}")

    # For each team: did model probability match outcome?
    # Simple check: rank correlation between model probs and actual finish
    actual_scores = {}
    for t in model_probs:
        if t == ACTUAL_RESULTS["champion"]:
            actual_scores[t] = 7
        elif t == ACTUAL_RESULTS["runner_up"]:
            actual_scores[t] = 6
        elif t == ACTUAL_RESULTS["third"]:
            actual_scores[t] = 5
        elif t == ACTUAL_RESULTS["fourth"]:
            actual_scores[t] = 4
        elif t in ACTUAL_RESULTS["quarterfinalists"]:
            actual_scores[t] = 3
        elif t in [ACTUAL_RESULTS["group_winners"][g] for g in ACTUAL_RESULTS["group_winners"]]:
            actual_scores[t] = 2
        elif t in [ACTUAL_RESULTS["group_runners"][g] for g in ACTUAL_RESULTS["group_runners"]]:
            actual_scores[t] = 1
        else:
            actual_scores[t] = 0

    common = [t for t in model_probs if t in actual_scores]
    mod_vals = [model_probs[t] for t in common]
    act_vals = [actual_scores[t] for t in common]

    from scipy.stats import spearmanr
    corr, pval = spearmanr(mod_vals, act_vals)
    print(f"\n  Spearman rank correlation (modelo vs resultado real): {corr:.3f} (p={pval:.4f})")

    # Brier score for champion
    brier = 0
    for t in model_probs:
        p = model_probs[t] / 100
        outcome = 1.0 if t == ACTUAL_RESULTS["champion"] else 0.0
        brier += (p - outcome) ** 2
    brier /= len(model_probs)
    print(f"  Brier score (campeon): {brier:.4f}")
    print(f"    (referencia: modelo uniforme 1/32 seria {((1/32 - 1)**2 + 31*(1/32)**2)/32:.4f})")

    # Top-N accuracy
    ranked_teams = sorted(model_probs.keys(), key=lambda t: -model_probs[t])
    for n in [3, 5, 8, 10]:
        top_n = set(ranked_teams[:n])
        if ACTUAL_RESULTS["champion"] in top_n:
            in_topn = "SI"
        else:
            in_topn = "NO"
        print(f"  Campeon real en top {n} del modelo: {in_topn}"
              f" (top {n}: {', '.join(ranked_teams[:n])})")


def main():
    print("=" * 85)
    print("  BACKTEST FUERA DE MUESTRA: MUNDIAL 2022")
    print("  Entrenamos con datos hasta nov 2022, simulamos, comparamos con realidad")
    print("=" * 85)

    df = load_results()
    cutoff = pd.Timestamp("2022-11-20")  # WC 2022 start

    print("\n  Entrenando modelo con datos hasta noviembre 2022...")
    model = train_backtest_model(df, cutoff)

    # Print strength ranking
    ranked = sorted(model.strength.items(), key=lambda x: -x[1])
    print(f"\n  Top 15 strength (pre-WC 2022):")
    for i, (t, s) in enumerate(ranked[:15], 1):
        actual = ""
        if t == "Argentina": actual = " <- CAMPEON"
        elif t == "France": actual = " <- FINAL"
        elif t == "Croatia": actual = " <- 3ro"
        elif t == "Morocco": actual = " <- 4to"
        print(f"    {i:>2}. {t:<20} {s:>+.3f}{actual}")

    results = run_backtest(model, n_sims=10000, seed=42)
    print_backtest_results(*results, model)


if __name__ == "__main__":
    main()
