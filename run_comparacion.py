"""Comparacion completa: Modelo vs Casas de Apuesta.

Compara probabilidades de campeon, estadisticas del torneo,
y genera un dashboard completo del Mundial 2026.
"""

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026
from run_wc2026 import (
    predict_lambdas_hybrid,
    sample_scoreline,
    sample_knockout_scoreline,
)


def load_market_probs(path=Path("data/betting_odds.csv")):
    """Convert American odds to fair probabilities (vig removed)."""
    df = pd.read_csv(path)
    df["implied_prob"] = 100.0 / (df["american_odds"] + 100.0)
    total = df["implied_prob"].sum()
    df["fair_prob"] = df["implied_prob"] / total  # remove vig
    return dict(zip(df["team"], df["fair_prob"] * 100))


def run_full_montecarlo(model, n_sims=10000, seed=42):
    """Run Monte Carlo collecting ALL statistics."""
    rng = np.random.default_rng(seed)

    # Match-level
    group_match_results = {}
    for gname, teams in GROUPS_2026.items():
        for i in range(4):
            for j in range(i + 1, 4):
                group_match_results[(gname, teams[i], teams[j])] = []

    # Group standings
    group_standings_count = {
        g: {t: Counter() for t in teams} for g, teams in GROUPS_2026.items()
    }

    # Tournament-level
    champion_count = Counter()
    finalist_count = Counter()
    semifinal_count = Counter()
    qf_count = Counter()

    # Tournament stats per simulation
    total_goals_per_sim = []
    total_matches_per_sim = []
    upsets_per_sim = []
    draws_group_per_sim = []
    extra_time_per_sim = []
    penalties_per_sim = []

    print(f"\n  Simulando {n_sims:,} mundiales...", flush=True)

    for sim in range(n_sims):
        if (sim + 1) % 2500 == 0:
            print(f"    {sim + 1:,}/{n_sims:,}...", flush=True)

        sim_goals = 0
        sim_matches = 0
        sim_upsets = 0
        sim_draws = 0
        sim_et = 0
        sim_pens = 0

        # --- Group stage ---
        standings = {}
        for gname, teams in GROUPS_2026.items():
            pts = [0] * 4
            gf = [0] * 4
            ga = [0] * 4

            for i in range(4):
                for j in range(i + 1, 4):
                    h, a = teams[i], teams[j]
                    lh, la = predict_lambdas_hybrid(model, h, a, neutral=True)
                    gh, ga_ = sample_scoreline(rng, lh, la)
                    group_match_results[(gname, h, a)].append((gh, ga_))

                    sim_goals += gh + ga_
                    sim_matches += 1
                    if gh == ga_:
                        sim_draws += 1

                    # Detect upset: weaker team wins
                    s_h = model.strength.get(h, 0)
                    s_a = model.strength.get(a, 0)
                    if (gh > ga_ and s_h < s_a - 0.3) or (ga_ > gh and s_a < s_h - 0.3):
                        sim_upsets += 1

                    gf[i] += gh; ga[i] += ga_
                    gf[j] += ga_; ga[j] += gh
                    if gh > ga_:
                        pts[i] += 3
                    elif gh == ga_:
                        pts[i] += 1; pts[j] += 1
                    else:
                        pts[j] += 3

            indices = list(range(4))
            indices.sort(
                key=lambda k: (pts[k], gf[k] - ga[k], gf[k], rng.random()),
                reverse=True,
            )
            standings[gname] = {
                "ranked": [teams[k] for k in indices],
                "pts": [pts[k] for k in indices],
                "gd": [gf[k] - ga[k] for k in indices],
                "gf": [gf[k] for k in indices],
            }
            for pos, k in enumerate(indices):
                group_standings_count[gname][teams[k]][pos + 1] += 1

        # --- Knockout ---
        thirds = []
        for g in sorted(standings):
            s = standings[g]
            thirds.append((g, s["ranked"][2], s["pts"][2], s["gd"][2], s["gf"][2]))
        thirds.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
        best_thirds = [t[1] for t in thirds[:8]]

        winners = {g: standings[g]["ranked"][0] for g in standings}
        runners = {g: standings[g]["ranked"][1] for g in standings}
        tl = best_thirds

        r32 = [
            (winners["A"], tl[0]), (runners["C"], runners["D"]),
            (winners["B"], tl[1]), (runners["E"], runners["F"]),
            (winners["G"], tl[2]), (runners["I"], runners["J"]),
            (winners["H"], tl[3]), (runners["K"], runners["L"]),
            (winners["C"], tl[4]), (runners["A"], runners["B"]),
            (winners["D"], tl[5]), (runners["G"], runners["H"]),
            (winners["I"], tl[6]), (winners["F"], runners["I"]),
            (winners["J"], tl[7]), (winners["L"], runners["K"]),
        ]

        def ko_match(h, a):
            nonlocal sim_goals, sim_matches, sim_upsets, sim_et, sim_pens
            lh, la = predict_lambdas_hybrid(model, h, a, neutral=True)
            gh, ga, et, pens = sample_knockout_scoreline(rng, lh, la)
            sim_goals += gh + ga
            sim_matches += 1
            if et:
                sim_et += 1
            if pens:
                sim_pens += 1
            s_h = model.strength.get(h, 0)
            s_a = model.strength.get(a, 0)
            winner = h if gh > ga else a
            loser = a if gh > ga else h
            if model.strength.get(loser, 0) > model.strength.get(winner, 0) + 0.3:
                sim_upsets += 1
            return winner

        r32w = [ko_match(h, a) for h, a in r32]
        r16 = [(r32w[i], r32w[i + 1]) for i in range(0, 16, 2)]
        r16w = [ko_match(h, a) for h, a in r16]
        qf = [(r16w[i], r16w[i + 1]) for i in range(0, 8, 2)]
        qfw = [ko_match(h, a) for h, a in qf]
        sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
        sfw = [ko_match(h, a) for h, a in sf]
        champ = ko_match(sfw[0], sfw[1])
        final_loser = sfw[1] if champ == sfw[0] else sfw[0]

        champion_count[champ] += 1
        finalist_count[champ] += 1
        finalist_count[final_loser] += 1
        for t in sfw:
            semifinal_count[t] += 1
        for i, (h, a) in enumerate(sf):
            loser = a if sfw[i] == h else h
            semifinal_count[loser] += 1
        for t in qfw:
            qf_count[t] += 1

        total_goals_per_sim.append(sim_goals)
        total_matches_per_sim.append(sim_matches)
        upsets_per_sim.append(sim_upsets)
        draws_group_per_sim.append(sim_draws)
        extra_time_per_sim.append(sim_et)
        penalties_per_sim.append(sim_pens)

    return {
        "group_match_results": group_match_results,
        "group_standings": group_standings_count,
        "champion": champion_count,
        "finalist": finalist_count,
        "semifinal": semifinal_count,
        "qf": qf_count,
        "total_goals": total_goals_per_sim,
        "total_matches": total_matches_per_sim,
        "upsets": upsets_per_sim,
        "draws_group": draws_group_per_sim,
        "extra_time": extra_time_per_sim,
        "penalties": penalties_per_sim,
        "n_sims": n_sims,
    }


def print_dashboard(data, model):
    """Print the full comparison dashboard."""
    N = data["n_sims"]
    market = load_market_probs()

    # =====================================================================
    # 1. CHAMPION COMPARISON: MODEL vs MARKET
    # =====================================================================
    print("\n" + "=" * 90)
    print("  1. CAMPEON: MODELO vs CASAS DE APUESTA")
    print("=" * 90)
    print(f"  {'#':<4} {'Equipo':<18} {'Modelo':>8} {'Mercado':>8} {'Diff':>8}"
          f" {'Odds':>8} {'Veredicto'}")
    print(f"  {'-' * 82}")

    model_probs = {t: c / N * 100 for t, c in data["champion"].items()}

    all_teams = sorted(
        set(list(model_probs.keys()) + list(market.keys())),
        key=lambda t: -(model_probs.get(t, 0) + market.get(t, 0)) / 2
    )

    for rank, team in enumerate(all_teams[:25], 1):
        p_mod = model_probs.get(team, 0)
        p_mkt = market.get(team, 0)
        diff = p_mod - p_mkt

        # Reconstruct American odds from fair prob
        if p_mkt > 0:
            odds_str = f"+{int(100 / (p_mkt / 100) - 100)}"
        else:
            odds_str = "N/A"

        if abs(diff) < 1.0:
            verdict = "OK"
        elif diff > 3.0:
            verdict = "SOBREVALORA"
        elif diff > 1.0:
            verdict = "algo alto"
        elif diff < -3.0:
            verdict = "INFRAVALORA"
        elif diff < -1.0:
            verdict = "algo bajo"
        else:
            verdict = "OK"

        print(f"  {rank:<4} {team:<18} {p_mod:>7.1f}% {p_mkt:>7.1f}% {diff:>+7.1f}%"
              f" {odds_str:>8} {verdict}")

    # Correlation
    common = [t for t in all_teams if t in model_probs and t in market]
    mod_vals = [model_probs.get(t, 0) for t in common]
    mkt_vals = [market.get(t, 0) for t in common]
    if len(mod_vals) > 2:
        corr = np.corrcoef(mod_vals, mkt_vals)[0, 1]
        mae = np.mean(np.abs(np.array(mod_vals) - np.array(mkt_vals)))
        print(f"\n  Correlacion modelo vs mercado: {corr:.3f}")
        print(f"  Error absoluto medio (MAE):    {mae:.2f} puntos porcentuales")

    # =====================================================================
    # 2. TOURNAMENT STATISTICS
    # =====================================================================
    print(f"\n\n{'=' * 90}")
    print("  2. ESTADISTICAS DEL TORNEO (promedios sobre {:,} simulaciones)".format(N))
    print("=" * 90)

    goals = np.array(data["total_goals"])
    matches = np.array(data["total_matches"])
    upsets = np.array(data["upsets"])
    draws = np.array(data["draws_group"])
    et = np.array(data["extra_time"])
    pens = np.array(data["penalties"])

    gpm = goals / matches

    print(f"\n  {'Metrica':<45} {'Promedio':>10} {'Min':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'-' * 82}")
    print(f"  {'Total de goles en el torneo':<45} {goals.mean():>10.0f}"
          f" {goals.min():>8} {goals.max():>8} {goals.std():>8.1f}")
    print(f"  {'Goles por partido':<45} {gpm.mean():>10.2f}"
          f" {gpm.min():>8.2f} {gpm.max():>8.2f} {gpm.std():>8.2f}")
    print(f"  {'Total de partidos':<45} {matches.mean():>10.0f}"
          f" {matches.min():>8} {matches.max():>8} {'-':>8}")
    print(f"  {'Empates en fase de grupos (de 72)':<45} {draws.mean():>10.1f}"
          f" {draws.min():>8} {draws.max():>8} {draws.std():>8.1f}")
    print(f"  {'Sorpresas (upset) por torneo':<45} {upsets.mean():>10.1f}"
          f" {upsets.min():>8} {upsets.max():>8} {upsets.std():>8.1f}")
    print(f"  {'Partidos con tiempo extra':<45} {et.mean():>10.1f}"
          f" {et.min():>8} {et.max():>8} {et.std():>8.1f}")
    print(f"  {'Partidos con penales':<45} {pens.mean():>10.1f}"
          f" {pens.min():>8} {pens.max():>8} {pens.std():>8.1f}")

    # Historical comparison
    print(f"\n  Referencia historica:")
    print(f"    WC 2022 (64 partidos): 172 goles, 2.69 gol/partido")
    print(f"    WC 2018 (64 partidos): 169 goles, 2.64 gol/partido")
    print(f"    WC 2014 (64 partidos): 171 goles, 2.67 gol/partido")
    print(f"    Nuestro modelo (104 partidos): {goals.mean():.0f} goles,"
          f" {gpm.mean():.2f} gol/partido")

    # =====================================================================
    # 3. MATCH-LEVEL: Group stage summary
    # =====================================================================
    print(f"\n\n{'=' * 90}")
    print("  3. FASE DE GRUPOS - PRONOSTICO POR PARTIDO")
    print("=" * 90)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        print(f"\n  --- Grupo {gname} ---")
        print(f"  {'Partido':<40} {'E[gol]':>10} {'Moda':>6}"
              f" {'1':>5} {'X':>5} {'2':>5} {'O2.5':>5}")
        print(f"  {'-' * 82}")

        for i in range(4):
            for j in range(i + 1, 4):
                key = (gname, teams[i], teams[j])
                results = data["group_match_results"][key]

                avg_h = np.mean([r[0] for r in results])
                avg_a = np.mean([r[1] for r in results])

                score_counts = Counter(results)
                mode = score_counts.most_common(1)[0][0]

                w1 = sum(1 for g1, g2 in results if g1 > g2) / N * 100
                dr = sum(1 for g1, g2 in results if g1 == g2) / N * 100
                w2 = sum(1 for g1, g2 in results if g1 < g2) / N * 100
                over25 = sum(1 for g1, g2 in results if g1 + g2 > 2.5) / N * 100

                match_name = f"{teams[i]} vs {teams[j]}"
                avg_str = f"{avg_h:.1f} - {avg_a:.1f}"
                mode_str = f"{mode[0]}-{mode[1]}"

                print(f"  {match_name:<40} {avg_str:>10} {mode_str:>6}"
                      f" {w1:>4.0f}% {dr:>4.0f}% {w2:>4.0f}% {over25:>4.0f}%")

    # =====================================================================
    # 4. GROUP STANDINGS
    # =====================================================================
    print(f"\n\n{'=' * 90}")
    print("  4. CLASIFICACION DE GRUPOS - Probabilidad por posicion")
    print("=" * 90)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        print(f"\n  Grupo {gname}")
        print(f"  {'Equipo':<22} {'1ro':>7} {'2do':>7} {'3ro':>7} {'4to':>7}"
              f"  {'Clasifica':>10}")
        print(f"  {'-' * 66}")

        team_data = []
        for t in teams:
            c = data["group_standings"][gname][t]
            p1 = c.get(1, 0) / N * 100
            p2 = c.get(2, 0) / N * 100
            p3 = c.get(3, 0) / N * 100
            p4 = c.get(4, 0) / N * 100
            team_data.append((t, p1, p2, p3, p4, p1 + p2))

        team_data.sort(key=lambda x: -x[5])
        for t, p1, p2, p3, p4, cl in team_data:
            print(f"  {t:<22} {p1:>6.1f}% {p2:>6.1f}% {p3:>6.1f}% {p4:>6.1f}%"
                  f"  {cl:>9.1f}%")

    # =====================================================================
    # 5. DEEP STATS
    # =====================================================================
    print(f"\n\n{'=' * 90}")
    print("  5. PROBABILIDADES AVANZADAS")
    print("=" * 90)

    # Finalist probabilities
    print(f"\n  {'Equipo':<18} {'Campeon':>9} {'Final':>9} {'Semi':>9} {'QF':>9}")
    print(f"  {'-' * 58}")
    for team, count in data["champion"].most_common(20):
        p_c = count / N * 100
        p_f = data["finalist"].get(team, 0) / N * 100
        p_s = data["semifinal"].get(team, 0) / N * 100
        p_q = data["qf"].get(team, 0) / N * 100
        print(f"  {team:<18} {p_c:>8.1f}% {p_f:>8.1f}% {p_s:>8.1f}% {p_q:>8.1f}%")

    # =====================================================================
    # 6. SCORE DISTRIBUTION
    # =====================================================================
    print(f"\n\n{'=' * 90}")
    print("  6. DISTRIBUCION DE MARCADORES (fase de grupos)")
    print("=" * 90)

    all_scores = []
    all_total_goals = []
    for results in data["group_match_results"].values():
        for gh, ga in results:
            all_scores.append((gh, ga))
            all_total_goals.append(gh + ga)

    score_dist = Counter(all_scores)
    total_group_matches = len(all_scores)

    print(f"\n  Marcadores mas frecuentes (de {total_group_matches:,} partidos simulados):")
    for (gh, ga), count in score_dist.most_common(15):
        pct = count / total_group_matches * 100
        bar = "#" * int(pct * 2)
        print(f"    {gh}-{ga}: {pct:>5.1f}%  {bar}")

    # Goals distribution
    goal_dist = Counter(all_total_goals)
    print(f"\n  Goles totales por partido:")
    for g in range(8):
        count = goal_dist.get(g, 0)
        pct = count / total_group_matches * 100
        bar = "#" * int(pct * 2)
        print(f"    {g} goles: {pct:>5.1f}%  {bar}")

    over25_pct = sum(1 for g in all_total_goals if g > 2.5) / total_group_matches * 100
    under25_pct = 100 - over25_pct
    btts_pct = sum(1 for gh, ga in all_scores if gh > 0 and ga > 0) / total_group_matches * 100

    print(f"\n  Over 2.5 goles:       {over25_pct:.1f}%")
    print(f"  Under 2.5 goles:      {under25_pct:.1f}%")
    print(f"  Ambos marcan (BTTS):  {btts_pct:.1f}%")

    # Clean sheets
    cs_pct = sum(1 for gh, ga in all_scores if gh == 0 or ga == 0) / total_group_matches * 100
    draw_pct = sum(1 for gh, ga in all_scores if gh == ga) / total_group_matches * 100
    print(f"  Porteria invicta:     {cs_pct:.1f}%")
    print(f"  Empates:              {draw_pct:.1f}%")

    # =====================================================================
    # 7. WHAT-IF: Biggest upset potential
    # =====================================================================
    print(f"\n\n{'=' * 90}")
    print("  7. PARTIDOS CON MAYOR POTENCIAL DE SORPRESA")
    print("=" * 90)

    upset_matches = []
    for (gname, home, away), results in data["group_match_results"].items():
        s_h = model.strength.get(home, 0)
        s_a = model.strength.get(away, 0)

        if s_h > s_a:
            fav, dog = home, away
            p_upset = sum(1 for g1, g2 in results if g1 < g2) / N * 100
        else:
            fav, dog = away, home
            p_upset = sum(1 for g1, g2 in results if g1 > g2) / N * 100

        diff = abs(s_h - s_a)
        if diff > 0.3 and p_upset > 5:
            upset_matches.append((gname, fav, dog, p_upset, diff))

    upset_matches.sort(key=lambda x: -x[3])
    print(f"\n  {'Grupo':<7} {'Favorito':<18} {'Outsider':<18} {'P(upset)':>9}")
    print(f"  {'-' * 56}")
    for gname, fav, dog, p_up, diff in upset_matches[:20]:
        print(f"  {gname:<7} {fav:<18} {dog:<18} {p_up:>8.1f}%")


def main():
    print("=" * 90)
    print("  MUNDIAL 2026 - DASHBOARD COMPLETO")
    print("  Modelo Hibrido (14 features) vs Casas de Apuesta")
    print("  10,000 simulaciones Monte Carlo")
    print("=" * 90)

    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    model = train_hybrid_model(df, cutoff)

    data = run_full_montecarlo(model, n_sims=10000, seed=42)
    print_dashboard(data, model)


if __name__ == "__main__":
    main()
