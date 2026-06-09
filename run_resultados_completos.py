"""Genera todos los resultados partido a partido del Mundial 2026."""
import numpy as np
import math
import pandas as pd
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026


def main():
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    print("Entrenando modelo...")
    model = train_hybrid_model(df, cutoff)

    rng = np.random.default_rng(42)
    N = 10000

    def plam(h, a):
        sh = model.strength.get(h, 0.0)
        sa = model.strength.get(a, 0.0)
        d = sh - sa
        return math.exp(0.25 + d / model.scale), math.exp(0.25 - d / model.scale)

    # Collect all group match results
    group_results = {}
    for gname, teams in GROUPS_2026.items():
        for i in range(4):
            for j in range(i + 1, 4):
                group_results[(gname, teams[i], teams[j])] = []

    group_pos = {g: {t: Counter() for t in teams} for g, teams in GROUPS_2026.items()}
    group_pts = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    group_gf = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    group_ga = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}

    champion_count = Counter()
    finalist_count = Counter()
    semi_count = Counter()

    print(f"  Simulando {N:,} mundiales...")

    for sim in range(N):
        if (sim + 1) % 2500 == 0:
            print(f"    {sim+1:,}/{N:,}...")

        standings = {}
        for gname, teams in GROUPS_2026.items():
            pts = [0] * 4; gf = [0] * 4; ga = [0] * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    lh, la = plam(teams[i], teams[j])
                    gh = int(rng.poisson(lh)); ga_ = int(rng.poisson(la))
                    group_results[(gname, teams[i], teams[j])].append((gh, ga_))
                    gf[i] += gh; ga[i] += ga_; gf[j] += ga_; ga[j] += gh
                    if gh > ga_: pts[i] += 3
                    elif gh == ga_: pts[i] += 1; pts[j] += 1
                    else: pts[j] += 3
            idx = list(range(4))
            idx.sort(key=lambda k: (pts[k], gf[k] - ga[k], gf[k], rng.random()), reverse=True)
            standings[gname] = {"ranked": [teams[k] for k in idx]}
            for pos, k in enumerate(idx):
                group_pos[gname][teams[k]][pos + 1] += 1
                group_pts[gname][teams[k]].append(pts[k])
                group_gf[gname][teams[k]].append(gf[k])
                group_ga[gname][teams[k]].append(ga[k])

        # Knockout
        thirds = [standings[g]["ranked"][2] for g in sorted(standings)]
        tl = thirds[:8]
        winners = {g: standings[g]["ranked"][0] for g in standings}
        runners = {g: standings[g]["ranked"][1] for g in standings}

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

        def ko(h, a):
            lh, la = plam(h, a)
            gh = int(rng.poisson(lh)); ga = int(rng.poisson(la))
            if gh == ga:
                if rng.random() < lh / (lh + la): gh += 1
                else: ga += 1
            return h if gh > ga else a

        r32w = [ko(h, a) for h, a in r32]
        r16 = [(r32w[i], r32w[i + 1]) for i in range(0, 16, 2)]
        r16w = [ko(h, a) for h, a in r16]
        qf = [(r16w[i], r16w[i + 1]) for i in range(0, 8, 2)]
        qfw = [ko(h, a) for h, a in qf]
        sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
        sfw = [ko(h, a) for h, a in sf]
        champ = ko(sfw[0], sfw[1])
        loser = sfw[1] if champ == sfw[0] else sfw[0]

        champion_count[champ] += 1
        finalist_count[champ] += 1
        finalist_count[loser] += 1
        for t in sfw:
            semi_count[t] += 1
        for i, (h, a) in enumerate(sf):
            sl = a if sfw[i] == h else h
            semi_count[sl] += 1

    # ===== PRINT RESULTS =====
    print("\n" + "=" * 100)
    print("  MUNDIAL 2026 - RESULTADOS COMPLETOS (10,000 simulaciones)")
    print("  Modelo: Hibrido 14 features | Elo corregido | Domestic discount")
    print("=" * 100)

    total_goals = 0
    total_matches = 0

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]

        print(f"\n  {'=' * 55} GRUPO {gname} {'=' * 55}")

        # Standings table
        team_data = []
        for t in teams:
            c = group_pos[gname][t]
            p1 = c.get(1, 0) / N * 100
            p2 = c.get(2, 0) / N * 100
            p3 = c.get(3, 0) / N * 100
            p4 = c.get(4, 0) / N * 100
            avg_pts = np.mean(group_pts[gname][t])
            avg_gf = np.mean(group_gf[gname][t])
            avg_ga = np.mean(group_ga[gname][t])
            classif = p1 + p2
            team_data.append((t, avg_pts, avg_gf, avg_ga, p1, p2, p3, p4, classif))
        team_data.sort(key=lambda x: -x[8])

        print(f"\n  Tabla esperada:")
        print(f"  {'Pos':<4} {'Equipo':<22} {'Pts':>5} {'GF':>5} {'GC':>5} {'GD':>5}"
              f"  {'1ro':>6} {'2do':>6} {'3ro':>6} {'4to':>6} {'Clasif':>7}")
        print(f"  {'-' * 95}")
        for rank, (t, pts, gf, ga, p1, p2, p3, p4, cl) in enumerate(team_data, 1):
            gd = gf - ga
            print(f"  {rank:<4} {t:<22} {pts:>5.1f} {gf:>5.1f} {ga:>5.1f} {gd:>+5.1f}"
                  f"  {p1:>5.1f}% {p2:>5.1f}% {p3:>5.1f}% {p4:>5.1f}% {cl:>6.1f}%")

        # Match results
        print(f"\n  Partidos:")
        print(f"  {'#':<3} {'Partido':<38} {'Pronostico':>12} {'Moda':>7}"
              f"  {'Top marcadores':<32} {'1':>5} {'X':>5} {'2':>5}")
        print(f"  {'-' * 110}")

        match_num = 0
        for i in range(4):
            for j in range(i + 1, 4):
                match_num += 1
                key = (gname, teams[i], teams[j])
                results = group_results[key]

                avg_h = np.mean([r[0] for r in results])
                avg_a = np.mean([r[1] for r in results])

                total_goals += sum(r[0] + r[1] for r in results)
                total_matches += N

                score_counts = Counter(results)
                top3 = score_counts.most_common(3)
                moda = top3[0]

                w1 = sum(1 for g1, g2 in results if g1 > g2) / N * 100
                dr = sum(1 for g1, g2 in results if g1 == g2) / N * 100
                w2 = sum(1 for g1, g2 in results if g1 < g2) / N * 100

                match = f"{teams[i]} vs {teams[j]}"
                avg_str = f"{avg_h:.1f} - {avg_a:.1f}"
                moda_str = f"{moda[0][0]}-{moda[0][1]}"
                top3_str = " ".join([f"{s[0]}-{s[1]}({c/N*100:.0f}%)" for s, c in top3])

                print(f"  {match_num:<3} {match:<38} {avg_str:>12} {moda_str:>7}"
                      f"  {top3_str:<32} {w1:>4.0f}% {dr:>4.0f}% {w2:>4.0f}%")

    # Goals summary
    avg_goals_per_match = total_goals / total_matches
    print(f"\n\n{'=' * 100}")
    print(f"  ESTADISTICAS DE GOLES")
    print(f"{'=' * 100}")
    print(f"  Goles promedio por partido (fase de grupos): {avg_goals_per_match:.2f}")
    print(f"  Total estimado fase de grupos: {avg_goals_per_match * 72:.0f} goles en 72 partidos")
    print(f"  Total estimado torneo completo: {avg_goals_per_match * 103:.0f} goles en 103 partidos")

    # Group winners summary
    print(f"\n\n{'=' * 100}")
    print(f"  CLASIFICADOS MAS PROBABLES POR GRUPO")
    print(f"{'=' * 100}")
    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        sorted_teams = sorted(teams, key=lambda t: -(group_pos[gname][t].get(1, 0) + group_pos[gname][t].get(2, 0)))
        t1, t2, t3 = sorted_teams[0], sorted_teams[1], sorted_teams[2]
        p1 = (group_pos[gname][t1].get(1, 0) + group_pos[gname][t1].get(2, 0)) / N * 100
        p2 = (group_pos[gname][t2].get(1, 0) + group_pos[gname][t2].get(2, 0)) / N * 100
        p3 = (group_pos[gname][t3].get(1, 0) + group_pos[gname][t3].get(2, 0)) / N * 100
        print(f"  Grupo {gname}: {t1:<18} ({p1:.0f}%) + {t2:<18} ({p2:.0f}%) | 3ro: {t3:<18} ({p3:.0f}%)")

    # Champion
    print(f"\n\n{'=' * 100}")
    print(f"  CAMPEON")
    print(f"{'=' * 100}")
    print(f"  {'#':<3} {'Equipo':<20} {'Campeon':>9} {'Final':>9} {'Semi':>9}")
    print(f"  {'-' * 53}")
    for rank, (team, count) in enumerate(champion_count.most_common(20), 1):
        pc = count / N * 100
        pf = finalist_count.get(team, 0) / N * 100
        ps = semi_count.get(team, 0) / N * 100
        bar = "#" * int(pc)
        print(f"  {rank:<3} {team:<20} {pc:>8.1f}% {pf:>8.1f}% {ps:>8.1f}%  {bar}")

    # Final matchups already in the montecarlo - just print champion


if __name__ == "__main__":
    main()
