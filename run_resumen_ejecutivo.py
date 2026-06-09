"""Resumen ejecutivo completo del Mundial 2026 - Un solo archivo para compartir."""
import numpy as np
import math
import pandas as pd
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026

MARKET = {
    "Spain": (450, 15.8), "France": (480, 14.9), "England": (700, 10.8),
    "Brazil": (850, 9.1), "Portugal": (850, 9.1), "Argentina": (900, 8.7),
    "Germany": (1400, 5.8), "Netherlands": (2000, 4.1), "Norway": (3500, 2.4),
    "Belgium": (4000, 2.1), "Colombia": (4000, 2.1), "Japan": (4500, 1.9),
    "Croatia": (5000, 1.7), "Switzerland": (5000, 1.7), "Morocco": (5000, 1.7),
    "Mexico": (5500, 1.5), "United States": (6000, 1.4), "Uruguay": (6500, 1.3),
    "Turkey": (8000, 1.1), "Ecuador": (10000, 0.9), "Senegal": (10000, 0.9),
    "South Korea": (15000, 0.6), "Canada": (22500, 0.4),
}


def main():
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    model = train_hybrid_model(df, cutoff)

    rng = np.random.default_rng(42)
    N = 10000

    def plam(h, a):
        sh = model.strength.get(h, 0.0)
        sa = model.strength.get(a, 0.0)
        d = sh - sa
        return math.exp(0.25 + d / model.scale), math.exp(0.25 - d / model.scale)

    # Collect everything
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
    qf_count = Counter()
    final_matchup = Counter()
    goal_sims = []

    print(f"Simulando {N:,} mundiales...", flush=True)

    for sim in range(N):
        if (sim + 1) % 2500 == 0:
            print(f"  {sim+1:,}/{N:,}...", flush=True)

        standings = {}
        sim_goals = [0]  # mutable for nested fn

        for gn, teams in GROUPS_2026.items():
            pts = [0]*4; gf = [0]*4; ga = [0]*4
            for i in range(4):
                for j in range(i + 1, 4):
                    lh, la = plam(teams[i], teams[j])
                    gh = int(rng.poisson(lh)); ga_ = int(rng.poisson(la))
                    group_results[(gn, teams[i], teams[j])].append((gh, ga_))
                    sim_goals[0] += gh + ga_
                    gf[i] += gh; ga[i] += ga_; gf[j] += ga_; ga[j] += gh
                    if gh > ga_: pts[i] += 3
                    elif gh == ga_: pts[i] += 1; pts[j] += 1
                    else: pts[j] += 3
            idx = list(range(4))
            idx.sort(key=lambda k: (pts[k], gf[k]-ga[k], gf[k], rng.random()), reverse=True)
            standings[gn] = {"ranked": [teams[k] for k in idx]}
            for pos, k in enumerate(idx):
                group_pos[gn][teams[k]][pos+1] += 1
                group_pts[gn][teams[k]].append(pts[k])
                group_gf[gn][teams[k]].append(gf[k])
                group_ga[gn][teams[k]].append(ga[k])

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
            sim_goals[0] += gh + ga
            if gh == ga:
                sim_goals[0] += 1
                if rng.random() < lh/(lh+la): gh += 1
                else: ga += 1
            return h if gh > ga else a

        r32w = [ko(h, a) for h, a in r32]
        r16 = [(r32w[i], r32w[i+1]) for i in range(0, 16, 2)]
        r16w = [ko(h, a) for h, a in r16]
        for t in r16w: qf_count[t] += 1
        qf = [(r16w[i], r16w[i+1]) for i in range(0, 8, 2)]
        qfw = [ko(h, a) for h, a in qf]
        sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
        sfw = [ko(h, a) for h, a in sf]
        champ = ko(sfw[0], sfw[1])
        loser = sfw[1] if champ == sfw[0] else sfw[0]

        champion_count[champ] += 1
        finalist_count[champ] += 1; finalist_count[loser] += 1
        final_matchup[tuple(sorted([sfw[0], sfw[1]]))] += 1
        for t in sfw: semi_count[t] += 1
        for i, (h, a) in enumerate(sf):
            sl = a if sfw[i] == h else h
            semi_count[sl] += 1
        goal_sims.append(sim_goals[0])

    # ===================================================================
    # WRITE EXECUTIVE SUMMARY
    # ===================================================================
    tg = np.array(goal_sims)

    lines = []
    def p(s=""): lines.append(s)

    p("=" * 100)
    p("  MUNDIAL 2026 - RESUMEN EJECUTIVO")
    p("  Modelo predictivo basado en 10,000 simulaciones Monte Carlo")
    p("=" * 100)
    p()
    p("  Fecha: 9 de junio de 2026 (2 dias antes del inicio)")
    p("  Modelo: Hibrido 14 features | Elo corregido | Poisson goal sampling")
    p("  Validacion: Backtest fuera de muestra 2010-2022, Brier Score < mercado")
    p()
    p()

    # --- SECTION 1: CHAMPION ---
    p("=" * 100)
    p("  1. QUIEN GANA EL MUNDIAL?")
    p("=" * 100)
    p()
    p(f"  {'#':<3} {'Equipo':<20} {'Campeon':>9} {'Final':>9} {'Semi':>9} {'QF':>9}  {'Mercado':>8} {'vs Mkt':>8}")
    p(f"  {'-' * 82}")
    for rank, (team, count) in enumerate(champion_count.most_common(20), 1):
        pc = count / N * 100
        pf = finalist_count.get(team, 0) / N * 100
        ps = semi_count.get(team, 0) / N * 100
        pq = qf_count.get(team, 0) / N * 100
        mkt = MARKET.get(team, (0, 0))[1]
        diff = pc - mkt if mkt > 0 else 0
        diff_str = f"{diff:+.1f}pp" if mkt > 0 else ""
        bar = "#" * int(pc)
        p(f"  {rank:<3} {team:<20} {pc:>8.1f}% {pf:>8.1f}% {ps:>8.1f}% {pq:>8.1f}%  {mkt:>7.1f}% {diff_str:>7}  {bar}")

    p()
    p("  LECTURA: France es nuestro favorito (19.6%), seguido de Spain (17.4%)")
    p("  y England (15.8%). El mercado tiene a Spain primero (15.8%).")
    p()

    # --- SECTION 2: FINALS ---
    p()
    p("=" * 100)
    p("  2. FINALES MAS PROBABLES")
    p("=" * 100)
    p()
    for (t1, t2), count in final_matchup.most_common(10):
        pct = count / N * 100
        bar = "#" * int(pct * 3)
        p(f"  {t1} vs {t2}: {pct:.1f}%  {bar}")

    # --- SECTION 3: GROUPS ---
    p()
    p()
    p("=" * 100)
    p("  3. FASE DE GRUPOS - TABLAS Y PARTIDOS")
    p("=" * 100)

    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        p(f"\n  {'='*45} GRUPO {gname} {'='*45}")

        # Table
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
            cl = p1 + p2
            team_data.append((t, avg_pts, avg_gf, avg_ga, p1, p2, p3, p4, cl))
        team_data.sort(key=lambda x: -x[8])

        p(f"\n  {'Pos':<4} {'Equipo':<22} {'Pts':>5} {'GF':>5} {'GC':>5} {'GD':>5}"
          f"  {'1ro':>6} {'2do':>6} {'3ro':>6} {'4to':>6} {'Clasif':>7}")
        p(f"  {'-' * 95}")
        for rank, (t, pts, gf, ga, p1, p2, p3, p4, cl) in enumerate(team_data, 1):
            gd = gf - ga
            p(f"  {rank:<4} {t:<22} {pts:>5.1f} {gf:>5.1f} {ga:>5.1f} {gd:>+5.1f}"
              f"  {p1:>5.1f}% {p2:>5.1f}% {p3:>5.1f}% {p4:>5.1f}% {cl:>6.1f}%")

        # Matches
        p()
        p(f"  {'Partido':<38} {'Pronostico':>12} {'Moda':>7}  {'1':>5} {'X':>5} {'2':>5}")
        p(f"  {'-' * 75}")

        for i in range(4):
            for j in range(i + 1, 4):
                key = (gname, teams[i], teams[j])
                results = group_results[key]
                avg_h = np.mean([r[0] for r in results])
                avg_a = np.mean([r[1] for r in results])
                score_counts = Counter(results)
                moda = score_counts.most_common(1)[0]
                w1 = sum(1 for g1, g2 in results if g1 > g2) / N * 100
                dr = sum(1 for g1, g2 in results if g1 == g2) / N * 100
                w2 = sum(1 for g1, g2 in results if g1 < g2) / N * 100
                match = f"{teams[i]} vs {teams[j]}"
                avg_str = f"{avg_h:.1f} - {avg_a:.1f}"
                moda_str = f"{moda[0][0]}-{moda[0][1]}"
                p(f"  {match:<38} {avg_str:>12} {moda_str:>7}  {w1:>4.0f}% {dr:>4.0f}% {w2:>4.0f}%")

    # --- SECTION 4: GOALS ---
    p()
    p()
    p("=" * 100)
    p("  4. ESTADISTICAS DE GOLES")
    p("=" * 100)
    p()
    p(f"  Total partidos:         103 (72 fase de grupos + 31 eliminatorias)")
    p(f"  Goles totales:          {tg.mean():.0f} promedio ({tg.min()}-{tg.max()} rango)")
    p(f"  Goles por partido:      {tg.mean()/103:.2f}")
    p(f"  Goles fase de grupos:   ~{tg.mean()/103*72:.0f} (en 72 partidos)")
    p()
    p(f"  Comparacion historica:")
    p(f"    WC 2022: 172 goles / 64 partidos = 2.69 goles/partido")
    p(f"    WC 2018: 169 goles / 64 partidos = 2.64 goles/partido")
    p(f"    WC 2026: {tg.mean():.0f} goles / 103 partidos = {tg.mean()/103:.2f} goles/partido")
    p()
    p(f"  Mas goles por partido que mundiales anteriores por la cantidad de")
    p(f"  equipos debiles (Haiti, Qatar, Curazao, Panama) vs selecciones top.")

    # --- SECTION 5: VALUE BETS ---
    p()
    p()
    p("=" * 100)
    p("  5. MODELO vs CASAS DE APUESTAS")
    p("=" * 100)
    p()
    p(f"  {'Equipo':<20} {'Modelo':>8} {'Mercado':>8} {'Diff':>8} {'Odds':>8} {'Senial':>12}")
    p(f"  {'-' * 68}")

    all_teams_sorted = sorted(
        MARKET.keys(),
        key=lambda t: -(champion_count.get(t, 0) / N * 100),
    )
    for team in all_teams_sorted:
        m_pct = champion_count.get(team, 0) / N * 100
        odds_am, mkt_pct = MARKET[team]
        diff = m_pct - mkt_pct
        if diff > 2:
            senial = "VALOR +"
        elif diff < -2:
            senial = "caro -"
        else:
            senial = "justo"
        p(f"  {team:<20} {m_pct:>7.1f}% {mkt_pct:>7.1f}% {diff:>+7.1f}pp {'+'+str(odds_am):>7} {senial:>12}")

    from scipy.stats import spearmanr
    m_vals = [champion_count.get(t, 0)/N*100 for t in all_teams_sorted]
    k_vals = [MARKET[t][1] for t in all_teams_sorted]
    sp, _ = spearmanr(m_vals, k_vals)
    mae = np.mean(np.abs(np.array(m_vals) - np.array(k_vals)))

    p()
    p(f"  Correlacion Spearman: {sp:.3f}")
    p(f"  Error promedio (MAE): {mae:.2f}pp")
    p()
    p(f"  APUESTAS DE VALOR (modelo > mercado):")
    for team in all_teams_sorted:
        m_pct = champion_count.get(team, 0) / N * 100
        odds_am, mkt_pct = MARKET[team]
        diff = m_pct - mkt_pct
        if diff > 2:
            p(f"    -> {team}: modelo {m_pct:.1f}% vs mercado {mkt_pct:.1f}% (odds +{odds_am})")
    p()
    p(f"  SOBREVALUADOS POR MERCADO (mercado > modelo):")
    for team in all_teams_sorted:
        m_pct = champion_count.get(team, 0) / N * 100
        odds_am, mkt_pct = MARKET[team]
        diff = m_pct - mkt_pct
        if diff < -2:
            p(f"    -> {team}: mercado {mkt_pct:.1f}% vs modelo {m_pct:.1f}%")

    # --- SECTION 6: METHODOLOGY ---
    p()
    p()
    p("=" * 100)
    p("  6. METODOLOGIA")
    p("=" * 100)
    p()
    p("  MODELO HIBRIDO DE 14 FEATURES EN 2 CAPAS:")
    p()
    p("  Layer 1 - Backtestable (calibrado con Ridge Regression):")
    p("    1. Elo rating (50%) - con time decay y ajuste por calidad del rival")
    p("    2. Valor de mercado del plantel (11%)")
    p("    3. Liga composite (7%) - con descuento domestico")
    p("    4. Momentum (14%) - ultimos 10 partidos, ponderado por Elo rival")
    p("    5. Solidez defensiva (18%) - goles recibidos ultimos 10 partidos")
    p("    6. Campeon defensor (8%) - penalidad historica por repetir")
    p()
    p("  Layer 2 - Especifico del torneo (priors informados):")
    p("    7. Odds de mercado (32%) - probabilidades implicitas de apuestas")
    p("    8. Edad del plantel (5%) - distancia a la edad optima (26.9)")
    p("    9. Antiguedad del DT (4%) - sweet spot 2-8 anios")
    p("   10. Ventaja local (2%) - sede USA/Mexico/Canada")
    p("   11. Poblacion (2%) - pool de talento")
    p("   12. Solidez defensiva L2 (6%)")
    p("   13. Diversidad diaspora (3%) - amplitud de reclutamiento")
    p("   14. Composicion historica (3%) - umbral de participaciones")
    p("   15. Frontrunner curse (5%) - penalidad al favorito extremo")
    p()
    p("  INNOVACIONES CLAVE:")
    p("    - Elo con time decay (0.94/anio): resultados viejos pesan menos")
    p("    - Elo ajustado por calidad: ganarle a Alemania > ganarle a Bolivia")
    p("    - K de clasificatorias reducido (30 vs 40): evita inflacion por volumen")
    p("    - Descuento domestico en liga: ser ingles en la PL no es lo mismo")
    p("      que ser marroqui en la PL (sesgo de seleccion)")
    p()
    p("  SIMULACION:")
    p("    - 10,000 mundiales simulados via Monte Carlo")
    p("    - Goles muestreados con distribucion de Poisson")
    p("    - Lambdas derivadas de la diferencia de fuerza entre equipos")
    p("    - Analogia: pricing de opciones financieras via Monte Carlo")
    p()
    p("  VALIDACION (backtest fuera de muestra 2010-2022):")
    p("    - Brier Score modelo: 0.0426 vs mercado: 0.0447 (modelo MEJOR)")
    p("    - Campeon real siempre en top 5 del modelo (4/4 mundiales)")
    p("    - Spearman promedio: ~0.50 (p<0.05)")
    p()
    p("=" * 100)
    p("  Generado el 9 de junio de 2026")
    p("  github.com/marti - Modelo predictivo WC 2026")
    p("=" * 100)

    # Write to file
    output = "\n".join(lines)
    with open("RESUMEN_EJECUTIVO_WC2026.txt", "w", encoding="utf-8") as f:
        f.write(output)
    print(output)
    print(f"\n\nArchivo guardado: RESUMEN_EJECUTIVO_WC2026.txt ({len(lines)} lineas)")


if __name__ == "__main__":
    main()
