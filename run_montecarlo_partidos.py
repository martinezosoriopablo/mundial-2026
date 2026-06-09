"""Monte Carlo por partido: marcador promedio, moda y probabilidades 1/X/2.

Simula 10,000 mundiales y extrae estadisticas a nivel de partido,
clasificacion de grupos, finales mas probables y campeon.

Analogia: pricing de opciones via Monte Carlo.
  10,000 trayectorias -> payoff promedio = precio justo
  10,000 torneos      -> marcador promedio = resultado esperado
"""

import numpy as np
import pandas as pd
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026
from run_wc2026 import (
    predict_lambdas_hybrid,
    sample_scoreline,
    sample_knockout_scoreline,
    simulate_group_stage,
    select_best_thirds,
    build_r32_bracket,
    simulate_knockout_round,
)


def run_match_level_montecarlo(model, n_sims=10000, seed=42):
    """Run Monte Carlo and collect match-level + tournament-level stats."""
    rng = np.random.default_rng(seed)

    # Group match results: (group, home, away) -> [(gh, ga), ...]
    group_match_results = {}
    for gname, teams in GROUPS_2026.items():
        for i in range(4):
            for j in range(i + 1, 4):
                group_match_results[(gname, teams[i], teams[j])] = []

    # Group standings: group -> {team -> Counter({1: n, 2: n, ...})}
    group_standings_count = {
        g: {t: Counter() for t in teams} for g, teams in GROUPS_2026.items()
    }

    # Knockout tracking
    final_matchup_count = Counter()
    champion_count = Counter()
    finalist_count = Counter()
    semifinal_count = Counter()

    print(f"\n  Simulando {n_sims:,} mundiales...", flush=True)

    for sim in range(n_sims):
        if (sim + 1) % 2500 == 0:
            print(f"    {sim + 1:,}/{n_sims:,}...", flush=True)

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
            lh, la = predict_lambdas_hybrid(model, h, a, neutral=True)
            gh, ga, _, _ = sample_knockout_scoreline(rng, lh, la)
            return h if gh > ga else a

        r32w = [ko_match(h, a) for h, a in r32]
        r16 = [(r32w[i], r32w[i + 1]) for i in range(0, 16, 2)]
        r16w = [ko_match(h, a) for h, a in r16]
        qf = [(r16w[i], r16w[i + 1]) for i in range(0, 8, 2)]
        qfw = [ko_match(h, a) for h, a in qf]
        sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
        sfw = [ko_match(h, a) for h, a in sf]
        champ = ko_match(sfw[0], sfw[1])
        final_loser = sfw[1] if champ == sfw[0] else sfw[0]

        final_matchup_count[tuple(sorted([sfw[0], sfw[1]]))] += 1
        champion_count[champ] += 1
        finalist_count[champ] += 1
        finalist_count[final_loser] += 1
        for t in sfw:
            semifinal_count[t] += 1
        for i, (h, a) in enumerate(sf):
            loser = a if sfw[i] == h else h
            semifinal_count[loser] += 1

    return (group_match_results, group_standings_count,
            final_matchup_count, champion_count, finalist_count,
            semifinal_count, n_sims)


def print_all_results(group_match_results, group_standings_count,
                      final_matchup_count, champion_count, finalist_count,
                      semifinal_count, n_sims):
    """Print complete results."""
    N = n_sims

    # === MATCH-LEVEL ===
    print("\n" + "=" * 90)
    print("  MUNDIAL 2026 - PRONOSTICO POR PARTIDO (Monte Carlo, {:,} simulaciones)".format(N))
    print("=" * 90)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        print(f"\n  --- Grupo {gname} ---")
        print(f"  {'Partido':<40} {'E[score]':>10} {'Moda':>8}"
              f" {'1':>5} {'X':>5} {'2':>5}")
        print(f"  {'-' * 78}")

        for i in range(4):
            for j in range(i + 1, 4):
                key = (gname, teams[i], teams[j])
                results = group_match_results[key]

                avg_h = np.mean([r[0] for r in results])
                avg_a = np.mean([r[1] for r in results])

                score_counts = Counter(results)
                mode_score = score_counts.most_common(1)[0]

                w1 = sum(1 for g1, g2 in results if g1 > g2) / N * 100
                dr = sum(1 for g1, g2 in results if g1 == g2) / N * 100
                w2 = sum(1 for g1, g2 in results if g1 < g2) / N * 100

                match_name = f"{teams[i]} vs {teams[j]}"
                avg_str = f"{avg_h:.1f} - {avg_a:.1f}"
                mode_str = f"{mode_score[0][0]}-{mode_score[0][1]}"

                print(f"  {match_name:<40} {avg_str:>10} {mode_str:>8}"
                      f" {w1:>4.0f}% {dr:>4.0f}% {w2:>4.0f}%")

    # === GROUP STANDINGS ===
    print(f"\n\n{'=' * 90}")
    print("  CLASIFICACION DE GRUPOS - Probabilidad por posicion")
    print("=" * 90)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        print(f"\n  Grupo {gname}")
        print(f"  {'Equipo':<22} {'1ro':>7} {'2do':>7} {'3ro':>7} {'4to':>7}"
              f"  {'Clasifica':>10}")
        print(f"  {'-' * 66}")

        team_data = []
        for t in teams:
            c = group_standings_count[gname][t]
            p1 = c.get(1, 0) / N * 100
            p2 = c.get(2, 0) / N * 100
            p3 = c.get(3, 0) / N * 100
            p4 = c.get(4, 0) / N * 100
            classif = p1 + p2
            team_data.append((t, p1, p2, p3, p4, classif))

        team_data.sort(key=lambda x: -x[5])
        for t, p1, p2, p3, p4, cl in team_data:
            print(f"  {t:<22} {p1:>6.1f}% {p2:>6.1f}% {p3:>6.1f}% {p4:>6.1f}%"
                  f"  {cl:>9.1f}%")

    # === FINALS ===
    print(f"\n\n{'=' * 90}")
    print("  FINALES MAS PROBABLES")
    print("=" * 90)
    for (t1, t2), count in final_matchup_count.most_common(15):
        print(f"  {t1} vs {t2}: {count / N * 100:.1f}%")

    # === CHAMPION ===
    print(f"\n{'=' * 90}")
    print("  CAMPEON")
    print("=" * 90)
    print(f"  {'Equipo':<18} {'Campeon':>9} {'Final':>9} {'Semi':>9}")
    print(f"  {'-' * 48}")
    for team, count in champion_count.most_common(15):
        p_c = count / N * 100
        p_f = finalist_count.get(team, 0) / N * 100
        p_s = semifinal_count.get(team, 0) / N * 100
        bar = "#" * int(p_c)
        print(f"  {team:<18} {p_c:>8.1f}% {p_f:>8.1f}% {p_s:>8.1f}%  {bar}")


def main():
    print("Cargando datos y entrenando modelo...")
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")

    print("Entrenando modelo hibrido (14 features)...")
    model = train_hybrid_model(df, cutoff)

    results = run_match_level_montecarlo(model, n_sims=10000, seed=42)
    print_all_results(*results)


if __name__ == "__main__":
    main()
