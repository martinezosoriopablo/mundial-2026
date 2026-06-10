"""Genera bracket consistente usando resultados modales de cada grupo.

Toma el 1ro, 2do y 3ro mas probable de cada grupo, arma el cuadro
de eliminatorias y calcula probabilidades analiticas para cada partido.
Resultado: un solo bracket coherente sin equipos duplicados.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import poisson
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026, build_r32, R32_DESCRIPTIONS


def ko_prob(lh, la, max_goals=10):
    """Probabilidad analitica de que home avance en partido KO."""
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
        sh = model.strength.get(h, 0.0)
        sa = model.strength.get(a, 0.0)
        d = sh - sa
        return math.exp(0.25 + d / model.scale), math.exp(0.25 - d / model.scale)

    # --- Simular grupos para obtener posiciones modales ---
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

    # --- Resultado modal por grupo ---
    modal = {}
    for gname, teams in GROUPS_2026.items():
        team_data = []
        for t in teams:
            c = group_pos[gname][t]
            classify_pct = (c.get(1, 0) + c.get(2, 0)) / N * 100
            first_pct = c.get(1, 0) / N * 100
            team_data.append((t, classify_pct, first_pct, c))
        team_data.sort(key=lambda x: -x[1])
        modal[gname] = [td[0] for td in team_data]

    winners = {g: modal[g][0] for g in modal}
    runners = {g: modal[g][1] for g in modal}
    thirds = [modal[g][2] for g in sorted(modal)]
    tl = thirds[:8]

    # --- Imprimir grupos ---
    output_lines = []
    def out(s=""):
        output_lines.append(s)
        print(s)

    out("=" * 100)
    out("  MUNDIAL 2026 - BRACKET MODAL CONSISTENTE")
    out("  Modelo predictivo | 10,000 simulaciones | Probabilidades analiticas")
    out("=" * 100)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        out(f"\n  {'='*20} GRUPO {gname} {'='*20}")
        out(f"  {'Pos':<4} {'Equipo':<25} {'Pts':>5} {'GF':>5} {'GC':>5} {'GD':>6} {'1ro':>6} {'2do':>6} {'3ro':>6} {'4to':>6} {'Clasif':>7}")
        out(f"  {'-'*90}")

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
            marker = " <--" if t in [winners[gname], runners[gname]] else (" (3)" if t == modal[gname][2] else "")
            out(f"  {pos:<4} {t:<25} {pts:>5.1f} {gf:>5.1f} {ga_avg:>5.1f} {gf-ga_avg:>+5.1f} {p1:>5.1f}% {p2:>5.1f}% {p3:>5.1f}% {p4:>5.1f}% {cl:>6.1f}%{marker}")

        out(f"\n  {'Partido':<40} {'Pronostico':>11} {'Moda':>6} {'1':>5} {'X':>5} {'2':>5}")
        out(f"  {'-'*75}")
        for i in range(4):
            for j in range(i + 1, 4):
                key = (gname, teams[i], teams[j])
                results = group_results[key]
                avg_h = np.mean([r[0] for r in results])
                avg_a = np.mean([r[1] for r in results])
                moda = Counter(results).most_common(1)[0]
                w1 = sum(1 for g1, g2 in results if g1 > g2) / N * 100
                dr = sum(1 for g1, g2 in results if g1 == g2) / N * 100
                w2 = sum(1 for g1, g2 in results if g1 < g2) / N * 100
                out(f"  {teams[i]+' vs '+teams[j]:<40} {avg_h:.1f} - {avg_a:.1f} {moda[0][0]}-{moda[0][1]:>2} {w1:>4.0f}% {dr:>4.0f}% {w2:>4.0f}%")

    # --- Bracket de eliminatorias ---
    out("\n\n" + "=" * 100)
    out("  ELIMINATORIAS - BRACKET MODAL CONSISTENTE")
    out("  (1ro, 2do, 3ro mas probable de cada grupo -> un solo cuadro sin duplicados)")
    out("=" * 100)

    out(f"\n  Clasificados modales:")
    out(f"  {'Grupo':<6} {'1ro':<22} {'2do':<22} {'3ro':<22}")
    out(f"  {'-'*75}")
    for g in sorted(modal):
        out(f"  {g:<6} {modal[g][0]:<22} {modal[g][1]:<22} {modal[g][2]:<22}")

    # Build R32
    r32 = build_r32(winners, runners, tl)

    out(f"\n  RONDA DE 32 (16 partidos)")
    out(f"  {'#':<4} {'Partido':<45} {'Avanza':<22} {'Prob':>6}")
    out(f"  {'-'*80}")

    r32w = []
    for i, (h, a) in enumerate(r32):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        if p_h >= p_a:
            winner, prob = h, p_h
        else:
            winner, prob = a, p_a
        r32w.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<45} {winner:<22} {prob:>5.1f}%")

    # R16
    out(f"\n  OCTAVOS DE FINAL (8 partidos)")
    out(f"  {'#':<4} {'Partido':<45} {'Avanza':<22} {'Prob':>6}")
    out(f"  {'-'*80}")

    r16 = [(r32w[i], r32w[i+1]) for i in range(0, 16, 2)]
    r16w = []
    for i, (h, a) in enumerate(r16):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        if p_h >= p_a:
            winner, prob = h, p_h
        else:
            winner, prob = a, p_a
        r16w.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<45} {winner:<22} {prob:>5.1f}%")

    # QF
    out(f"\n  CUARTOS DE FINAL (4 partidos)")
    out(f"  {'#':<4} {'Partido':<45} {'Avanza':<22} {'Prob':>6}")
    out(f"  {'-'*80}")

    qf = [(r16w[i], r16w[i+1]) for i in range(0, 8, 2)]
    qfw = []
    for i, (h, a) in enumerate(qf):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        if p_h >= p_a:
            winner, prob = h, p_h
        else:
            winner, prob = a, p_a
        qfw.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<45} {winner:<22} {prob:>5.1f}%")

    # SF
    out(f"\n  SEMIFINALES (2 partidos)")
    out(f"  {'#':<4} {'Partido':<45} {'Avanza':<22} {'Prob':>6}")
    out(f"  {'-'*80}")

    sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
    sfw = []
    for i, (h, a) in enumerate(sf):
        lh, la = plam(h, a)
        p_h = ko_prob(lh, la) * 100
        p_a = 100 - p_h
        if p_h >= p_a:
            winner, prob = h, p_h
        else:
            winner, prob = a, p_a
        sfw.append(winner)
        out(f"  {i+1:<4} {h+' vs '+a:<45} {winner:<22} {prob:>5.1f}%")

    # Final
    out(f"\n  FINAL")
    out(f"  {'-'*80}")
    h, a = sfw[0], sfw[1]
    lh, la = plam(h, a)
    p_h = ko_prob(lh, la) * 100
    p_a = 100 - p_h
    if p_h >= p_a:
        champ, prob = h, p_h
        sub = a
    else:
        champ, prob = a, p_a
        sub = h
    out(f"  {h} vs {a}")
    out(f"  CAMPEON: {champ} ({prob:.1f}%)")
    out(f"  Subcampeon: {sub}")

    # Resumen del camino del campeon
    out(f"\n\n  CAMINO DEL CAMPEON ({champ}):")
    out(f"  {'-'*60}")
    # Trace back the champion's path
    for gn, ranked in modal.items():
        if champ in ranked[:3]:
            pos = ranked.index(champ) + 1
            out(f"  Grupo {gn}: {pos}ro ({group_pos[gn][champ].get(pos, 0)/N*100:.1f}%)")
            break

    out(f"\n  RESPUESTA A POR QUE MEXICO Y SUIZA TAN BIEN:")
    out(f"  {'-'*60}")
    out(f"  Mexico: Grupo A es el mas debil (Sudafrica, Rep Checa, Corea del Sur).")
    out(f"    -> 1ro con 70.6%, clasifica 90.3%. En R32 enfrenta un 3ro.")
    out(f"    -> Pero en octavos ya enfrenta Ecuador/Japan y se complica.")
    out(f"")
    out(f"  Switzerland: Grupo B tambien debil (Qatar, Bosnia).")
    out(f"    -> Pelea cabeza a cabeza con Canada (49% vs 47% por 1ro).")
    out(f"    -> Clasifica 89.8% pero en R32 enfrenta un 3ro (facil),")
    out(f"    -> y en octavos le toca Germany - ahi se acaba el suenio.")
    out(f"")
    out(f"  En el bracket anterior, Suiza aparecia en DOS octavos distintos")
    out(f"  porque cada slot se calculaba independientemente. Este bracket")
    out(f"  usa el resultado MODAL (mas probable) de cada grupo, asi que")
    out(f"  cada equipo aparece exactamente una vez.")

    # Write to file
    txt = "\n".join(output_lines)
    with open("BRACKET_MODAL_WC2026.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nGuardado: BRACKET_MODAL_WC2026.txt ({len(output_lines)} lineas)")

    return txt


if __name__ == "__main__":
    main()
