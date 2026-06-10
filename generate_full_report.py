"""Generate comprehensive World Cup 2026 prediction report with ALL statistics."""
import math
import numpy as np
import pandas as pd
from scipy.stats import poisson
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026, build_r32
from run_wc2026 import predict_lambdas_hybrid
from run_montecarlo import run_montecarlo


def ko_prob(lh, la, max_goals=10):
    p_h_win = 0.0
    p_draw = 0.0
    for gh in range(max_goals):
        for ga in range(max_goals):
            p = poisson.pmf(gh, lh) * poisson.pmf(ga, la)
            if gh > ga:
                p_h_win += p
            elif gh == ga:
                p_draw += p
    p_h_pens = lh / (lh + la)
    return p_h_win + p_draw * p_h_pens


def main():
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    print("Entrenando modelo...")
    model = train_hybrid_model(df, cutoff)

    def plam(h, a):
        return predict_lambdas_hybrid(model, h, a, neutral=True)

    # --- Simulate groups ---
    rng = np.random.default_rng(42)
    N = 10000
    group_pos = {g: {t: Counter() for t in teams} for g, teams in GROUPS_2026.items()}
    group_pts = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    group_gf = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    group_ga = {g: {t: [] for t in teams} for g, teams in GROUPS_2026.items()}
    group_results = {}
    for gname, teams in GROUPS_2026.items():
        for i in range(4):
            for j in range(i + 1, 4):
                group_results[(gname, teams[i], teams[j])] = []

    print(f"Simulando {N:,} fases de grupos...")
    for sim in range(N):
        if (sim + 1) % 2500 == 0:
            print(f"  {sim+1:,}/{N:,}...")
        for gn, teams in GROUPS_2026.items():
            pts = [0]*4; gf = [0]*4; ga = [0]*4
            for i in range(4):
                for j in range(i + 1, 4):
                    lh, la = plam(teams[i], teams[j])
                    gh = int(rng.poisson(lh)); ga_ = int(rng.poisson(la))
                    group_results[(gn, teams[i], teams[j])].append((gh, ga_))
                    gf[i] += gh; ga[i] += ga_; gf[j] += ga_; ga[j] += gh
                    if gh > ga_: pts[i] += 3
                    elif gh == ga_: pts[i] += 1; pts[j] += 1
                    else: pts[j] += 3
            idx = list(range(4))
            idx.sort(key=lambda k: (pts[k], gf[k]-ga[k], gf[k], rng.random()), reverse=True)
            for pos, k in enumerate(idx):
                group_pos[gn][teams[k]][pos + 1] += 1
                group_pts[gn][teams[k]].append(pts[k])
                group_gf[gn][teams[k]].append(gf[k])
                group_ga[gn][teams[k]].append(ga[k])

    # --- Modal results ---
    modal = {}
    for gname, teams in GROUPS_2026.items():
        team_data = []
        for t in teams:
            c = group_pos[gname][t]
            classify_pct = (c.get(1, 0) + c.get(2, 0)) / N * 100
            team_data.append((t, classify_pct, c))
        team_data.sort(key=lambda x: -x[1])
        modal[gname] = [td[0] for td in team_data]

    winners = {g: modal[g][0] for g in modal}
    runners = {g: modal[g][1] for g in modal}
    thirds_all = []
    for g in sorted(modal):
        t = modal[g][2]
        avg_pts = np.mean(group_pts[g][t])
        avg_gd = np.mean(group_gf[g][t]) - np.mean(group_ga[g][t])
        avg_gf = np.mean(group_gf[g][t])
        c = group_pos[g][t]
        p3 = c.get(3, 0) / N * 100
        thirds_all.append((g, t, avg_pts, avg_gd, avg_gf, p3))
    thirds_all.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = thirds_all[:8]
    tl = [t[1] for t in best_thirds]

    # Build bracket
    r32 = build_r32(winners, runners, tl)

    # --- BUILD REPORT ---
    lines = []
    def out(s=""):
        lines.append(s)

    out("=" * 110)
    out("  REPORTE COMPLETO - PREDICCION MUNDIAL FIFA 2026")
    out("  Modelo: Hybrid L1(Elo+Value+League+Defense) + L2(Age+Coach+Host+Pop+Diversity+Composition+SOS)")
    out("  Match: Dixon-Coles + H2H | Scale={:.2f} | Rho={:.2f} | Knockout upset sigma=0.15".format(model.scale, model.rho))
    out(f"  Simulaciones: {N:,} | Fecha: 2026-06-10")
    out("=" * 110)

    # =====================================================================
    # SECTION 1: ALL GROUP MATCHES (72 matches)
    # =====================================================================
    out("\n")
    out("=" * 110)
    out("  SECCION 1: TODOS LOS PARTIDOS DE FASE DE GRUPOS (72 partidos)")
    out("=" * 110)

    total_goals_tournament = 0.0
    total_matches_groups = 0
    all_matches = []  # (home, away, avg_h, avg_a, moda, w1, dr, w2, group)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        out(f"\n  {'='*25} GRUPO {gname} {'='*25}")
        out(f"  {'#':<4} {'Partido':<45} {'Pron':>11} {'Moda':>7} {'1':>6} {'X':>6} {'2':>6}")
        out(f"  {'-'*90}")

        match_num = 0
        for i in range(4):
            for j in range(i + 1, 4):
                match_num += 1
                total_matches_groups += 1
                key = (gname, teams[i], teams[j])
                results = group_results[key]
                avg_h = np.mean([r[0] for r in results])
                avg_a = np.mean([r[1] for r in results])
                total_goals_tournament += avg_h + avg_a
                moda = Counter(results).most_common(1)[0]
                w1 = sum(1 for g1, g2 in results if g1 > g2) / N * 100
                dr = sum(1 for g1, g2 in results if g1 == g2) / N * 100
                w2 = sum(1 for g1, g2 in results if g1 < g2) / N * 100
                label = f"{teams[i]} vs {teams[j]}"
                out(f"  {match_num:<4} {label:<45} {avg_h:.1f} - {avg_a:.1f} {moda[0][0]}-{moda[0][1]:>2} {w1:>5.1f}% {dr:>5.1f}% {w2:>5.1f}%")
                all_matches.append((teams[i], teams[j], avg_h, avg_a, moda[0], w1, dr, w2, gname))

    # =====================================================================
    # SECTION 2: GROUP STANDINGS
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 2: TABLAS DE POSICIONES DE TODOS LOS GRUPOS")
    out("=" * 110)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        out(f"\n  {'='*25} GRUPO {gname} {'='*25}")
        out(f"  {'Pos':<5} {'Equipo':<25} {'Pts':>6} {'GF':>6} {'GC':>6} {'GD':>7} {'1ro':>7} {'2do':>7} {'3ro':>7} {'4to':>7} {'Clasif':>8}")
        out(f"  {'-'*100}")

        team_data = []
        for t in teams:
            c = group_pos[gname][t]
            p1 = c.get(1, 0) / N * 100
            p2 = c.get(2, 0) / N * 100
            p3 = c.get(3, 0) / N * 100
            p4 = c.get(4, 0) / N * 100
            pts = np.mean(group_pts[gname][t])
            gf = np.mean(group_gf[gname][t])
            ga_avg = np.mean(group_ga[gname][t])
            team_data.append((t, pts, gf, ga_avg, p1, p2, p3, p4, p1 + p2))
        team_data.sort(key=lambda x: -x[8])

        for pos, (t, pts, gf, ga_avg, p1, p2, p3, p4, cl) in enumerate(team_data, 1):
            marker = ""
            if t == winners[gname]:
                marker = " << 1ro"
            elif t == runners[gname]:
                marker = " << 2do"
            elif t == modal[gname][2]:
                marker = " (3ro)"
            out(f"  {pos:<5} {t:<25} {pts:>5.1f} {gf:>6.1f} {ga_avg:>6.1f} {gf-ga_avg:>+6.1f} {p1:>6.1f}% {p2:>6.1f}% {p3:>6.1f}% {p4:>6.1f}% {cl:>7.1f}%{marker}")

    # =====================================================================
    # SECTION 3: GROUP WINNERS SUMMARY
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 3: RESUMEN - GANADORES DE GRUPO")
    out("=" * 110)
    out(f"\n  {'Grupo':<7} {'1ro':<25} {'Prob 1ro':>9} {'2do':<25} {'Prob 2do':>9} {'3ro':<25}")
    out(f"  {'-'*105}")
    for g in sorted(modal):
        t1 = modal[g][0]
        t2 = modal[g][1]
        t3 = modal[g][2]
        p1 = group_pos[g][t1].get(1, 0) / N * 100
        p2 = group_pos[g][t2].get(2, 0) / N * 100
        out(f"  {g:<7} {t1:<25} {p1:>8.1f}% {t2:<25} {p2:>8.1f}% {t3:<25}")

    # =====================================================================
    # SECTION 4: BEST THIRD-PLACED TEAMS
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 4: MEJORES TERCEROS (8 de 12 clasifican)")
    out("=" * 110)
    out(f"\n  {'#':<4} {'Grupo':<7} {'Equipo':<25} {'Pts prom':>9} {'GD prom':>9} {'GF prom':>9} {'Prob 3ro':>9} {'Clasifica':>10}")
    out(f"  {'-'*85}")
    for i, (g, t, pts, gd, gf, p3) in enumerate(thirds_all, 1):
        clasif = "SI" if i <= 8 else "NO"
        out(f"  {i:<4} {g:<7} {t:<25} {pts:>8.1f} {gd:>+8.1f} {gf:>8.1f} {p3:>8.1f}% {'  >> ' + clasif:>10}")

    # =====================================================================
    # SECTION 5: FULL KNOCKOUT BRACKET
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 5: ELIMINATORIAS COMPLETAS")
    out("=" * 110)

    # R32
    out(f"\n  --- RONDA DE 32 (16 partidos) ---")
    out(f"  {'#':<4} {'Partido':<50} {'Avanza':<22} {'Prob':>7} {'Lambda H':>9} {'Lambda A':>9}")
    out(f"  {'-'*105}")

    r32w = []
    for i, (h, a) in enumerate(r32):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        winner = h if p_h >= p_a else a
        prob = max(p_h, p_a)
        r32w.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<50} {winner:<22} {prob:>6.1f}% {lh:>8.2f} {la:>8.2f}")

    # R16
    out(f"\n  --- OCTAVOS DE FINAL (8 partidos) ---")
    out(f"  {'#':<4} {'Partido':<50} {'Avanza':<22} {'Prob':>7} {'Lambda H':>9} {'Lambda A':>9}")
    out(f"  {'-'*105}")

    r16 = [(r32w[i], r32w[i+1]) for i in range(0, 16, 2)]
    r16w = []
    for i, (h, a) in enumerate(r16):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        winner = h if p_h >= p_a else a
        prob = max(p_h, p_a)
        r16w.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<50} {winner:<22} {prob:>6.1f}% {lh:>8.2f} {la:>8.2f}")

    # QF
    out(f"\n  --- CUARTOS DE FINAL (4 partidos) ---")
    out(f"  {'#':<4} {'Partido':<50} {'Avanza':<22} {'Prob':>7} {'Lambda H':>9} {'Lambda A':>9}")
    out(f"  {'-'*105}")

    qf = [(r16w[i], r16w[i+1]) for i in range(0, 8, 2)]
    qfw = []
    for i, (h, a) in enumerate(qf):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        winner = h if p_h >= p_a else a
        prob = max(p_h, p_a)
        qfw.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<50} {winner:<22} {prob:>6.1f}% {lh:>8.2f} {la:>8.2f}")

    # SF
    out(f"\n  --- SEMIFINALES (2 partidos) ---")
    out(f"  {'#':<4} {'Partido':<50} {'Avanza':<22} {'Prob':>7} {'Lambda H':>9} {'Lambda A':>9}")
    out(f"  {'-'*105}")

    sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
    sfw = []
    for i, (h, a) in enumerate(sf):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        winner = h if p_h >= p_a else a
        prob = max(p_h, p_a)
        sfw.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<50} {winner:<22} {prob:>6.1f}% {lh:>8.2f} {la:>8.2f}")

    # Final
    out(f"\n  --- FINAL ---")
    out(f"  {'-'*105}")
    h, a = sfw[0], sfw[1]
    lh, la = plam(h, a)
    p_h = ko_prob(lh, la) * 100
    p_a = 100 - p_h
    champ = h if p_h >= p_a else a
    sub = a if champ == h else h
    prob_champ = max(p_h, p_a)
    out(f"  {h} vs {a}")
    out(f"  Lambda {h}: {lh:.2f}  |  Lambda {a}: {la:.2f}")
    out(f"")
    out(f"  *** CAMPEON: {champ} ({prob_champ:.1f}%) ***")
    out(f"  Subcampeon: {sub}")

    # =====================================================================
    # SECTION 6: VISUAL BRACKET
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 6: BRACKET VISUAL")
    out("=" * 110)
    out("")
    out("  RONDA 32              OCTAVOS              CUARTOS         SEMIS          FINAL")
    out("")
    # Top half
    out(f"  {r32[0][0]:<20} ---+")
    out(f"  {r32[0][1]:<20}    |--- {r16[0][0]:<16} ---+")
    out(f"  {r32[1][0]:<20} ---+                       |")
    out(f"  {r32[1][1]:<20}                            |--- {qf[0][0]:<14} ---+")
    out(f"  {r32[2][0]:<20} ---+                       |                       |")
    out(f"  {r32[2][1]:<20}    |--- {r16[1][0]:<16} ---+                       |")
    out(f"  {r32[3][0]:<20} ---+                                               |")
    out(f"  {r32[3][1]:<20}                                                    |--- {sf[0][0]:<12} ---+")
    out(f"  {r32[4][0]:<20} ---+                                               |                       |")
    out(f"  {r32[4][1]:<20}    |--- {r16[2][0]:<16} ---+                       |                       |")
    out(f"  {r32[5][0]:<20} ---+                       |                       |                       |")
    out(f"  {r32[5][1]:<20}                            |--- {qf[1][0]:<14} ---+                       |")
    out(f"  {r32[6][0]:<20} ---+                       |                                               |")
    out(f"  {r32[6][1]:<20}    |--- {r16[3][0]:<16} ---+                                               |")
    out(f"  {r32[7][0]:<20} ---+                                                                       |")
    out(f"  {r32[7][1]:<20}                                                                            |")
    out(f"                                                                                              |")
    out(f"                                                                             CAMPEON: {champ}")
    out(f"                                                                                              |")
    out(f"  {r32[8][0]:<20} ---+                                                                       |")
    out(f"  {r32[8][1]:<20}    |--- {r16[4][0]:<16} ---+                                               |")
    out(f"  {r32[9][0]:<20} ---+                       |                                               |")
    out(f"  {r32[9][1]:<20}                            |--- {qf[2][0]:<14} ---+                       |")
    out(f"  {r32[10][0]:<20} ---+                       |                       |                       |")
    out(f"  {r32[10][1]:<20}    |--- {r16[5][0]:<16} ---+                       |                       |")
    out(f"  {r32[11][0]:<20} ---+                                               |--- {sf[1][0]:<12} ---+")
    out(f"  {r32[11][1]:<20}                                                    |")
    out(f"  {r32[12][0]:<20} ---+                                               |")
    out(f"  {r32[12][1]:<20}    |--- {r16[6][0]:<16} ---+                       |")
    out(f"  {r32[13][0]:<20} ---+                       |                       |")
    out(f"  {r32[13][1]:<20}                            |--- {qf[3][0]:<14} ---+")
    out(f"  {r32[14][0]:<20} ---+                       |")
    out(f"  {r32[14][1]:<20}    |--- {r16[7][0]:<16} ---+")
    out(f"  {r32[15][0]:<20} ---+")
    out(f"  {r32[15][1]:<20}")

    # =====================================================================
    # SECTION 7: STATISTICS
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 7: ESTADISTICAS GENERALES")
    out("=" * 110)

    # Goals per group
    out(f"\n  --- Goles por grupo (promedios de {N:,} simulaciones) ---")
    out(f"  {'Grupo':<7} {'Goles totales':>14} {'Goles/partido':>14} {'Goleador del grupo':<25} {'GF prom':>8}")
    out(f"  {'-'*75}")

    group_goals_list = []
    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        total_g = 0
        best_scorer = ""
        best_gf = 0
        for t in teams:
            gf = np.mean(group_gf[gname][t])
            total_g += gf
            if gf > best_gf:
                best_gf = gf
                best_scorer = t
        gpp = total_g / 6
        group_goals_list.append((gname, total_g, gpp, best_scorer, best_gf))
        out(f"  {gname:<7} {total_g:>13.1f} {gpp:>13.1f} {best_scorer:<25} {best_gf:>7.1f}")

    total_goals_groups = sum(g[1] for g in group_goals_list)
    out(f"\n  Total goles fase de grupos: {total_goals_groups:.0f}")
    out(f"  Promedio goles/partido (grupos): {total_goals_groups/72:.2f}")
    out(f"  Grupo con mas goles: {max(group_goals_list, key=lambda x: x[1])[0]} ({max(group_goals_list, key=lambda x: x[1])[1]:.1f})")
    out(f"  Grupo con menos goles: {min(group_goals_list, key=lambda x: x[1])[0]} ({min(group_goals_list, key=lambda x: x[1])[1]:.1f})")

    # Most lopsided matches
    out(f"\n  --- Top 10 partidos mas desequilibrados ---")
    out(f"  {'#':<4} {'Partido':<45} {'Prob favorito':>13} {'Pron':>12}")
    out(f"  {'-'*78}")
    all_matches.sort(key=lambda x: -max(x[5], x[7]))
    for i, (h, a, avg_h, avg_a, moda, w1, dr, w2, g) in enumerate(all_matches[:10], 1):
        fav_prob = max(w1, w2)
        out(f"  {i:<4} {h+' vs '+a+f' (Gr.{g})':<45} {fav_prob:>12.1f}% {avg_h:.1f} - {avg_a:.1f}")

    # Most even matches
    out(f"\n  --- Top 10 partidos mas parejos ---")
    out(f"  {'#':<4} {'Partido':<45} {'1':>6} {'X':>6} {'2':>6}")
    out(f"  {'-'*72}")
    all_matches.sort(key=lambda x: abs(x[5] - x[7]))
    for i, (h, a, avg_h, avg_a, moda, w1, dr, w2, g) in enumerate(all_matches[:10], 1):
        out(f"  {i:<4} {h+' vs '+a+f' (Gr.{g})':<45} {w1:>5.1f}% {dr:>5.1f}% {w2:>5.1f}%")

    # Highest draw probability
    out(f"\n  --- Top 10 partidos con mayor probabilidad de empate ---")
    out(f"  {'#':<4} {'Partido':<45} {'Empate':>8} {'Pron':>12}")
    out(f"  {'-'*72}")
    all_matches.sort(key=lambda x: -x[6])
    for i, (h, a, avg_h, avg_a, moda, w1, dr, w2, g) in enumerate(all_matches[:10], 1):
        out(f"  {i:<4} {h+' vs '+a+f' (Gr.{g})':<45} {dr:>7.1f}% {avg_h:.1f} - {avg_a:.1f}")

    # Highest scoring matches
    out(f"\n  --- Top 10 partidos con mas goles esperados ---")
    out(f"  {'#':<4} {'Partido':<45} {'Goles esp':>10} {'Pron':>12}")
    out(f"  {'-'*75}")
    all_matches.sort(key=lambda x: -(x[2] + x[3]))
    for i, (h, a, avg_h, avg_a, moda, w1, dr, w2, g) in enumerate(all_matches[:10], 1):
        out(f"  {i:<4} {h+' vs '+a+f' (Gr.{g})':<45} {avg_h+avg_a:>9.1f} {avg_h:.1f} - {avg_a:.1f}")

    # Lowest scoring matches
    out(f"\n  --- Top 10 partidos con menos goles esperados ---")
    out(f"  {'#':<4} {'Partido':<45} {'Goles esp':>10} {'Pron':>12}")
    out(f"  {'-'*75}")
    all_matches.sort(key=lambda x: (x[2] + x[3]))
    for i, (h, a, avg_h, avg_a, moda, w1, dr, w2, g) in enumerate(all_matches[:10], 1):
        out(f"  {i:<4} {h+' vs '+a+f' (Gr.{g})':<45} {avg_h+avg_a:>9.1f} {avg_h:.1f} - {avg_a:.1f}")

    # =====================================================================
    # SECTION 8: ELO RANKINGS
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 8: RANKING DE FUERZA DEL MODELO (48 equipos)")
    out("=" * 110)

    wc_teams = set()
    for teams in GROUPS_2026.values():
        wc_teams.update(teams)

    strengths = [(t, model.strength.get(t, 0.0)) for t in wc_teams]
    strengths.sort(key=lambda x: -x[1])
    out(f"\n  {'#':<4} {'Equipo':<25} {'Fuerza':>8} {'Grupo':>7}")
    out(f"  {'-'*48}")
    team_to_group = {}
    for g, teams in GROUPS_2026.items():
        for t in teams:
            team_to_group[t] = g
    for i, (t, s) in enumerate(strengths, 1):
        out(f"  {i:<4} {t:<25} {s:>7.3f} {team_to_group[t]:>7}")

    # =====================================================================
    # SECTION 9: MONTE CARLO CHAMPION PROBABILITIES
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 9: PROBABILIDADES DE CAMPEON (Monte Carlo {:,} sims, Poisson + upset factor)".format(N))
    out("=" * 110)

    champ_count, final_count, semi_count, qf_count, n_mc = run_montecarlo(
        model, predict_lambdas_hybrid, "Full Report",
        n_sims=N, seed=42, method="poisson"
    )

    # Market odds for comparison
    market_odds = {
        "Spain": 18.2, "France": 17.5, "England": 13.3, "Brazil": 10.5,
        "Portugal": 10.0, "Argentina": 9.5, "Germany": 6.7, "Netherlands": 5.6,
        "Belgium": 4.3, "Norway": 2.9, "Japan": 2.8, "Colombia": 2.4,
        "United States": 2.0, "Morocco": 1.8, "Mexico": 1.8, "Uruguay": 1.8,
    }

    out(f"\n  {'#':<4} {'Equipo':<22} {'Campeon':>9} {'Final':>9} {'Semi':>9} {'QF':>9} {'Mercado':>9} {'Diff':>8}")
    out(f"  {'-'*75}")
    sorted_champs = sorted(champ_count.keys(), key=lambda t: -champ_count[t])
    for i, team in enumerate(sorted_champs[:30], 1):
        p_champ = champ_count.get(team, 0) / n_mc * 100
        p_final = final_count.get(team, 0) / n_mc * 100
        p_semi = semi_count.get(team, 0) / n_mc * 100
        p_qf = qf_count.get(team, 0) / n_mc * 100
        mkt = market_odds.get(team, 0)
        diff_str = f"{p_champ - mkt:+.1f}pp" if mkt > 0 else ""
        out(f"  {i:<4} {team:<22} {p_champ:>8.1f}% {p_final:>8.1f}% {p_semi:>8.1f}% {p_qf:>8.1f}% {mkt:>8.1f}% {diff_str:>8}")

    # Summary stats
    top6 = sorted(champ_count.values(), reverse=True)[:6]
    top6_conc = sum(top6) / n_mc * 100
    unique_champs = sum(1 for v in champ_count.values() if v > 0)
    total_se = sum((champ_count.get(t, 0)/n_mc*100 - market_odds[t])**2 for t in market_odds)
    rmse = math.sqrt(total_se / len(market_odds))

    out(f"\n  Top 6 concentracion: {top6_conc:.1f}%")
    out(f"  Equipos distintos campeon: {unique_champs}")
    out(f"  RMSE vs mercado (FanDuel): {rmse:.2f}pp")

    # =====================================================================
    # SECTION 10: CHAMPION PATH
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 10: CAMINO DEL CAMPEON ({})".format(champ))
    out("=" * 110)

    for gn, ranked in modal.items():
        if champ in ranked[:3]:
            pos = ranked.index(champ) + 1
            c = group_pos[gn][champ]
            p = c.get(pos, 0) / N * 100
            pts = np.mean(group_pts[gn][champ])
            gf = np.mean(group_gf[gn][champ])
            ga_avg = np.mean(group_ga[gn][champ])
            out(f"\n  Fase de grupos: Grupo {gn} - {pos}ro ({p:.1f}%)")
            out(f"  Pts promedio: {pts:.1f} | GF: {gf:.1f} | GC: {ga_avg:.1f} | GD: {gf-ga_avg:+.1f}")
            break

    # Trace knockout path
    path_rounds = ["R32", "Octavos", "Cuartos", "Semifinal", "Final"]
    path_opponents = []
    # R32
    for i, (h, a) in enumerate(r32):
        if h == champ or a == champ:
            opp = a if h == champ else h
            lh, la = plam(champ, opp)
            p = ko_prob(lh, la) * 100
            path_opponents.append(("Ronda de 32", opp, p))
    # R16
    for i, (h, a) in enumerate(r16):
        if r16w[i] == champ:
            opp = a if h == champ else h
            lh, la = plam(champ, opp)
            p = ko_prob(lh, la) * 100
            path_opponents.append(("Octavos", opp, p))
    # QF
    for i, (h, a) in enumerate(qf):
        if qfw[i] == champ:
            opp = a if h == champ else h
            lh, la = plam(champ, opp)
            p = ko_prob(lh, la) * 100
            path_opponents.append(("Cuartos", opp, p))
    # SF
    for i, (h, a) in enumerate(sf):
        if sfw[i] == champ:
            opp = a if h == champ else h
            lh, la = plam(champ, opp)
            p = ko_prob(lh, la) * 100
            path_opponents.append(("Semifinal", opp, p))
    # Final
    opp = sub
    lh, la = plam(champ, opp)
    p = ko_prob(lh, la) * 100
    path_opponents.append(("Final", opp, p))

    out(f"\n  {'Ronda':<15} {'Rival':<25} {'Prob avanzar':>12}")
    out(f"  {'-'*55}")
    for ronda, opp, prob in path_opponents:
        out(f"  {ronda:<15} {opp:<25} {prob:>11.1f}%")

    # =====================================================================
    # SECTION 11: MODEL INFO
    # =====================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 11: INFORMACION DEL MODELO")
    out("=" * 110)
    out(f"\n  Tipo: Hybrid Multi-Feature + Dixon-Coles + H2H + Knockout Upset Factor")
    out(f"  RPS calibracion: {0.1586:.4f}")
    out(f"  Scale: {model.scale:.2f} (floor=3.50, calibrado vs mercado)")
    out(f"  Home advantage: {model.home_adv:.2f}")
    out(f"  Dixon-Coles rho: {model.rho:.2f}")
    out(f"  H2H weight: {model.h2h_weight} ({len(model.h2h)} pares)")
    out(f"  Knockout upset sigma: 0.15")
    out(f"")
    out(f"  Features L1 (backtestable, Ridge regression):")
    l1w = {k.replace('L1_', ''): v for k, v in model.weights.items() if k.startswith('L1_')}
    for k, v in l1w.items():
        out(f"    - {k:<12} {v:.0%}")
    out(f"")
    out(f"  Features L2 (tournament-specific, informed priors):")
    l2w = {k: v for k, v in model.weights.items() if not k.startswith('L1_')}
    for k, v in l2w.items():
        out(f"    - {k:<14} {v:.0%}")
    out(f"")
    out(f"  Elo system: quality_exponent=0.5, decay=0.94/year, K: WC=60, Cont=50, Qual=30")
    out(f"")
    out(f"  Removed features (worsened OOS RPS):")
    out(f"    - Market odds (circular benchmark)")
    out(f"    - Defending champion (n=1 per tournament)")
    out(f"    - Momentum (correlated with Elo)")
    out(f"    - Frontrunner curse (n=7 sample)")
    out(f"")
    out(f"  Backtesting: Walk-forward (train before tournament, predict during)")
    out(f"  Tournaments validated: WC 2010/14/18/22, Euro 2012/16/20/24,")
    out(f"                         Copa America 2011/15/16/19/21/24")
    out(f"")
    out(f"  Calibracion torneo: scale + upset_sigma grid-searched vs FanDuel odds Jun 2026")

    # Write file
    txt = "\n".join(lines)
    outfile = "REPORTE_COMPLETO_WC2026.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nGuardado: {outfile} ({len(lines)} lineas)")

    return txt


if __name__ == "__main__":
    main()
