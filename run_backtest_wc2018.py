"""Backtest fuera de muestra: WC 2018.

Entrena el modelo con datos SOLO hasta junio 2018, simula el mundial 2018
con Monte Carlo, y compara predicciones vs resultados reales.
"""

import math
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
from src.data_loader import load_results
from src.strength_elo import compute_elo_ratings
from src.strength_hybrid import (
    load_squad_values, _zscore_dict, _blend_all_features,
    _batch_poisson_1x2, HybridModel,
)
from src.features import (
    compute_momentum, compute_defensive_strength, compute_league_composite,
    compute_defending_champion_feature,
)
from run_wc2026 import sample_scoreline, sample_knockout_scoreline

# =========================================================================
# WC 2018 DATA
# =========================================================================

GROUPS_2018 = {
    "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
    "B": ["Portugal", "Spain", "Morocco", "Iran"],
    "C": ["France", "Australia", "Peru", "Denmark"],
    "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
    "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
    "F": ["Germany", "Mexico", "Sweden", "South Korea"],
    "G": ["Belgium", "Panama", "Tunisia", "England"],
    "H": ["Poland", "Senegal", "Colombia", "Japan"],
}

ACTUAL_2018 = {
    "champion": "France",
    "runner_up": "Croatia",
    "third": "Belgium",
    "fourth": "England",
    "semifinalists": ["France", "Croatia", "Belgium", "England"],
    "quarterfinalists": ["France", "Croatia", "Belgium", "England",
                         "Uruguay", "Brazil", "Russia", "Sweden"],
    "group_winners": {
        "A": "Uruguay", "B": "Spain", "C": "France", "D": "Croatia",
        "E": "Brazil", "F": "Sweden", "G": "Belgium", "H": "Colombia",
    },
    "group_runners": {
        "A": "Russia", "B": "Portugal", "C": "Denmark", "D": "Argentina",
        "E": "Switzerland", "F": "Mexico", "G": "England", "H": "Japan",
    },
    "group_eliminated": [
        "Saudi Arabia", "Egypt", "Morocco", "Iran", "Australia", "Peru",
        "Iceland", "Nigeria", "Costa Rica", "Serbia", "Germany",
        "South Korea", "Panama", "Tunisia", "Poland", "Senegal",
    ],
    "upsets": [
        "Mexico beat Germany (groups)",
        "South Korea beat Germany (groups)",
        "Germany eliminated in groups (defending champion!)",
        "Russia QF (host, ranked 70th)",
        "Croatia 1st over Argentina (groups)",
        "Japan almost beat Belgium (R16, 2-0 lead)",
    ],
}

# Pre-tournament betting odds 2018 (approximate market consensus)
MARKET_2018 = {
    "Brazil": 18.0, "Germany": 14.0, "France": 12.0, "Spain": 11.0,
    "Argentina": 8.0, "Belgium": 7.0, "England": 5.0, "Portugal": 4.0,
    "Uruguay": 3.0, "Colombia": 2.5, "Croatia": 2.0, "Poland": 1.5,
    "Mexico": 1.0, "Denmark": 0.8, "Switzerland": 0.8, "Peru": 0.5,
    "Sweden": 0.5, "Senegal": 0.4, "Nigeria": 0.3, "Serbia": 0.3,
    "Japan": 0.3, "Russia": 0.5, "Iceland": 0.2, "Australia": 0.2,
    "Iran": 0.1, "Egypt": 0.2, "Morocco": 0.2, "Costa Rica": 0.1,
    "South Korea": 0.1, "Tunisia": 0.1, "Panama": 0.05, "Saudi Arabia": 0.05,
}

# =========================================================================
# MODEL TRAINING (reuse same logic as WC2022 backtest)
# =========================================================================

def train_backtest_model(df, cutoff):
    """Train Layer 1 model (backtestable features only)."""
    from sklearn.linear_model import RidgeCV

    ratings = compute_elo_ratings(df, cutoff)
    squad_values = load_squad_values()
    league_composite = compute_league_composite()
    momentum_norm = _zscore_dict(compute_momentum(df, cutoff, n_matches=10, elo_ratings=ratings))
    defense_norm = _zscore_dict(compute_defensive_strength(df, cutoff, n_matches=10))

    all_teams = set(ratings.keys()) | set(squad_values.keys()) | set(momentum_norm.keys())
    defending = compute_defending_champion_feature(cutoff, all_teams)

    elo_vals = np.array(list(ratings.values()))
    elo_mean, elo_std = elo_vals.mean(), elo_vals.std()
    elo_norm = {t: (ratings.get(t, 1500.0) - elo_mean) / elo_std for t in all_teams}

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

    outcomes = np.array([
        0 if int(r["home_score"]) > int(r["away_score"])
        else (1 if int(r["home_score"]) == int(r["away_score"]) else 2)
        for _, r in cal_data.iterrows()
    ])
    eye3 = np.eye(3)

    w_arr = np.array([model_w.get(nm, 0.0) for nm in cal_feat_names])
    s_h = feat_h @ w_arr
    s_a = feat_a @ w_arr

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

    l1_features = {
        "elo": elo_norm, "value": squad_values, "league": league_composite,
        "momentum": momentum_norm, "defense": defense_norm, "champion": defending,
    }
    model_blend = _blend_all_features(l1_features, model_w)

    return HybridModel(model_blend, best_ha, best_scale, model_w)


# =========================================================================
# SIMULATION
# =========================================================================

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


def sim_tournament_2018(model, rng):
    """Simulate WC 2018 bracket (32 teams, 8 groups of 4, top 2 advance)."""
    standings = sim_group_stage(model, GROUPS_2018, rng)

    winners = {g: standings[g][0] for g in standings}
    runners = {g: standings[g][1] for g in standings}

    # WC 2018 actual R16 bracket structure
    r16 = [
        (winners["A"], runners["B"]),  # Uruguay vs Portugal
        (winners["C"], runners["D"]),  # France vs Argentina
        (winners["B"], runners["A"]),  # Spain vs Russia
        (winners["D"], runners["C"]),  # Croatia vs Denmark
        (winners["E"], runners["F"]),  # Brazil vs Mexico
        (winners["G"], runners["H"]),  # Belgium vs Japan
        (winners["F"], runners["E"]),  # Sweden vs Switzerland
        (winners["H"], runners["G"]),  # Colombia vs England
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
    group_pos_count = {g: {t: Counter() for t in teams}
                       for g, teams in GROUPS_2018.items()}

    print(f"\n  Simulando {n_sims:,} WC 2018...", flush=True)
    for sim in range(n_sims):
        if (sim + 1) % 2500 == 0:
            print(f"    {sim + 1:,}/{n_sims:,}...", flush=True)

        champ, finalist, sf_losers, qfw, standings = sim_tournament_2018(model, rng)

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
            for pos, t in enumerate(standings[g]):
                group_pos_count[g][t][pos + 1] += 1

    return (champion_count, finalist_count, sf_count, qf_count,
            group_pos_count, n_sims)


# =========================================================================
# RESULTS
# =========================================================================

def print_results(champion_count, finalist_count, sf_count, qf_count,
                  group_pos_count, n_sims, model):
    N = n_sims

    # === 1. CHAMPION ===
    print(f"\n{'=' * 85}")
    print(f"  BACKTEST WC 2018 - MODELO vs REALIDAD ({N:,} simulaciones)")
    print(f"{'=' * 85}")

    print(f"\n  {'#':<4} {'Equipo':<18} {'Modelo':>8} {'Mercado':>8} {'Real':>12}")
    print(f"  {'-' * 70}")

    model_probs = {t: c / N * 100 for t, c in champion_count.items()}
    all_teams = sorted(
        set(list(model_probs.keys()) + list(MARKET_2018.keys())),
        key=lambda t: -(model_probs.get(t, 0) + MARKET_2018.get(t, 0)) / 2
    )

    for rank, team in enumerate(all_teams[:20], 1):
        p_mod = model_probs.get(team, 0)
        p_mkt = MARKET_2018.get(team, 0)
        actual = ""
        if team == ACTUAL_2018["champion"]:
            actual = "CAMPEON"
        elif team == ACTUAL_2018["runner_up"]:
            actual = "FINAL"
        elif team == ACTUAL_2018["third"]:
            actual = "3ro"
        elif team == ACTUAL_2018["fourth"]:
            actual = "4to (SF)"
        elif team in ACTUAL_2018["quarterfinalists"]:
            actual = "QF"
        elif team in ACTUAL_2018["group_eliminated"]:
            actual = "Eliminado"
        print(f"  {rank:<4} {team:<18} {p_mod:>7.1f}% {p_mkt:>7.1f}%  {actual:>12}")

    # === 2. VEREDICTO ===
    champ = ACTUAL_2018["champion"]
    p_champ = model_probs.get(champ, 0)
    rank_champ = sorted(model_probs.keys(), key=lambda t: -model_probs[t]).index(champ) + 1

    print(f"\n  {'=' * 60}")
    print(f"  VEREDICTO: El modelo le daba a {champ} (campeon real):")
    print(f"    Probabilidad: {p_champ:.1f}%")
    print(f"    Ranking: #{rank_champ}")
    print(f"    Mercado le daba: {MARKET_2018.get(champ, 0):.1f}%")

    top4_actual = ACTUAL_2018["semifinalists"]
    print(f"\n  Top 4 real: {', '.join(top4_actual)}")
    for t in top4_actual:
        p_c = model_probs.get(t, 0)
        p_s = sf_count.get(t, 0) / N * 100
        print(f"    {t:<18} Campeon={p_c:.1f}%  Semi={p_s:.1f}%")

    # === 3. GROUPS ===
    print(f"\n\n{'=' * 85}")
    print(f"  FASE DE GRUPOS: PREDICCION vs REALIDAD")
    print(f"{'=' * 85}")

    correct_winners = 0
    correct_qualifiers = 0

    for g in sorted(GROUPS_2018.keys()):
        teams = GROUPS_2018[g]
        actual_w = ACTUAL_2018["group_winners"][g]
        actual_r = ACTUAL_2018["group_runners"][g]

        print(f"\n  Grupo {g}  (Real: 1ro={actual_w}, 2do={actual_r})")
        print(f"  {'Equipo':<18} {'1ro':>7} {'2do':>7} {'3ro':>7} {'4to':>7}  {'Real'}")
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
            if t == actual_w: actual_pos = "<- 1ro"
            elif t == actual_r: actual_pos = "<- 2do"
            elif t in ACTUAL_2018["group_eliminated"]: actual_pos = "   elim"
            print(f"  {t:<18} {p1:>6.1f}% {p2:>6.1f}% {p3:>6.1f}% {p4:>6.1f}%  {actual_pos}")

        pred_winner = max(teams, key=lambda t: group_pos_count[g][t].get(1, 0))
        if pred_winner == actual_w:
            correct_winners += 1
        pred_top2 = sorted(teams, key=lambda t: -(group_pos_count[g][t].get(1, 0) + group_pos_count[g][t].get(2, 0)))[:2]
        if set(pred_top2) == {actual_w, actual_r}:
            correct_qualifiers += 1

    print(f"\n  Precision grupos:")
    print(f"    Ganador de grupo correcto:    {correct_winners}/8"
          f" ({correct_winners / 8 * 100:.0f}%)")
    print(f"    Top 2 clasificados correcto:  {correct_qualifiers}/8"
          f" ({correct_qualifiers / 8 * 100:.0f}%)")

    # === 4. SURPRISES ===
    print(f"\n\n{'=' * 85}")
    print(f"  SORPRESAS DEL MUNDIAL 2018: LAS PREDIJO EL MODELO?")
    print(f"{'=' * 85}")

    surprises = [
        ("Germany eliminada en grupos (campeon defensor!)", "Germany",
         (group_pos_count["F"]["Germany"].get(3, 0) +
          group_pos_count["F"]["Germany"].get(4, 0)) / N * 100),
        ("Sweden 1ro grupo F (sobre Germany, Mexico)", "Sweden",
         group_pos_count["F"]["Sweden"].get(1, 0) / N * 100),
        ("Russia QF (host, ranked ~70)", "Russia",
         qf_count.get("Russia", 0) / N * 100),
        ("Croatia 1ro grupo D (sobre Argentina)", "Croatia",
         group_pos_count["D"]["Croatia"].get(1, 0) / N * 100),
        ("Croatia finalista", "Croatia",
         finalist_count.get("Croatia", 0) / N * 100),
        ("France campeon (3er favorito)", "France",
         champion_count.get("France", 0) / N * 100),
        ("Belgium 3ro (sobre England, Brazil)", "Belgium",
         sf_count.get("Belgium", 0) / N * 100),
        ("Japan 2do grupo H (sobre Senegal, Poland)", "Japan",
         group_pos_count["H"]["Japan"].get(2, 0) / N * 100),
    ]

    print(f"\n  {'Sorpresa':<55} {'P(modelo)':>10}")
    print(f"  {'-' * 68}")
    for desc, team, prob in surprises:
        label = "Capturo!" if prob > 10 else ("Parcial" if prob > 3 else "NO")
        print(f"  {desc:<55} {prob:>9.1f}%  {label}")

    # === 5. CALIBRATION ===
    print(f"\n\n{'=' * 85}")
    print(f"  CALIBRACION DEL MODELO")
    print(f"{'=' * 85}")

    actual_scores = {}
    for t in model_probs:
        if t == ACTUAL_2018["champion"]: actual_scores[t] = 7
        elif t == ACTUAL_2018["runner_up"]: actual_scores[t] = 6
        elif t == ACTUAL_2018["third"]: actual_scores[t] = 5
        elif t == ACTUAL_2018["fourth"]: actual_scores[t] = 4
        elif t in ACTUAL_2018["quarterfinalists"]: actual_scores[t] = 3
        elif t in [ACTUAL_2018["group_winners"][g] for g in ACTUAL_2018["group_winners"]]: actual_scores[t] = 2
        elif t in [ACTUAL_2018["group_runners"][g] for g in ACTUAL_2018["group_runners"]]: actual_scores[t] = 1
        else: actual_scores[t] = 0

    common = [t for t in model_probs if t in actual_scores]
    mod_vals = [model_probs[t] for t in common]
    act_vals = [actual_scores[t] for t in common]
    corr, pval = spearmanr(mod_vals, act_vals)
    print(f"\n  Spearman rank correlation (modelo vs resultado real): {corr:.3f} (p={pval:.4f})")

    brier = sum((model_probs.get(t, 0) / 100 - (1.0 if t == ACTUAL_2018["champion"] else 0.0)) ** 2
                for t in model_probs) / len(model_probs)
    print(f"  Brier score (campeon): {brier:.4f}")
    print(f"    (referencia: modelo uniforme 1/32 seria {((1/32 - 1)**2 + 31*(1/32)**2)/32:.4f})")

    ranked_teams = sorted(model_probs.keys(), key=lambda t: -model_probs[t])
    for n in [3, 5, 8]:
        top_n = ranked_teams[:n]
        hit = "SI" if ACTUAL_2018["champion"] in top_n else "NO"
        print(f"  Campeon real en top {n}: {hit} (top {n}: {', '.join(top_n)})")


def main():
    print("=" * 85)
    print("  BACKTEST FUERA DE MUESTRA: MUNDIAL 2018")
    print("  Entrenamos con datos hasta junio 2018, simulamos, comparamos con realidad")
    print("=" * 85)

    df = load_results()
    cutoff = pd.Timestamp("2018-06-14")  # WC 2018 start

    print("\n  Entrenando modelo con datos hasta junio 2018...")
    model = train_backtest_model(df, cutoff)

    ranked = sorted(model.strength.items(), key=lambda x: -x[1])
    print(f"\n  Top 15 strength (pre-WC 2018):")
    for i, (t, s) in enumerate(ranked[:15], 1):
        actual = ""
        if t == "France": actual = " <- CAMPEON"
        elif t == "Croatia": actual = " <- FINAL"
        elif t == "Belgium": actual = " <- 3ro"
        elif t == "England": actual = " <- 4to"
        print(f"    {i:>2}. {t:<20} {s:>+.3f}{actual}")

    results = run_backtest(model, n_sims=10000, seed=42)
    print_results(*results, model)


if __name__ == "__main__":
    main()
