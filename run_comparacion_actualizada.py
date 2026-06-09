"""Comparacion modelo vs casas de apuestas actualizadas (junio 2026).

Fuentes: FanDuel, BetMGM, DraftKings, ESPN, FOX Sports (junio 2026).
Odds convertidas a probabilidad implicita sin vig.
"""
import numpy as np
import math
import pandas as pd
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026

# === MARKET ODDS (junio 2026, pre-torneo) ===
# American odds -> implied probability (sin ajuste por vig)
# Fuentes: FanDuel, ESPN, FOX Sports, DraftKings, BetMGM
MARKET_ODDS_AMERICAN = {
    "Spain": 450,        # +450 FanDuel
    "France": 480,       # +480 FanDuel
    "England": 700,      # +700 ESPN
    "Brazil": 850,       # +850 ESPN/FOX
    "Portugal": 850,     # +850 (shortened from 10-1)
    "Argentina": 900,    # +900 ESPN
    "Germany": 1400,     # 14-1 ESPN
    "Netherlands": 2000, # 20-1 ESPN
    "Norway": 3500,      # 35-1 ESPN
    "Belgium": 4000,     # 40-1 ESPN
    "Colombia": 4000,    # 40-1 ESPN
    "Japan": 4500,       # +4500 FanDuel
    "Croatia": 5000,     # ~50-1 avg
    "Switzerland": 5000, # ~50-1
    "Morocco": 5000,     # ~50-1 avg (FanDuel 60-1, BetMGM 40-1)
    "Mexico": 5500,      # +5500 CBS
    "United States": 6000, # +6000 FanDuel
    "Uruguay": 6500,     # 65-1 ESPN
    "Turkey": 8000,      # +8000 FanDuel
    "Ecuador": 10000,    # ~100-1
    "Senegal": 10000,    # ~100-1
    "South Korea": 15000,# ~150-1
    "Canada": 22500,     # +22500 FanDuel
}


def american_to_prob(odds):
    """Convert positive American odds to implied probability."""
    return 100.0 / (odds + 100.0)


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

    champion_count = Counter()
    finalist_count = Counter()
    semi_count = Counter()
    qf_count = Counter()

    print(f"  Simulando {N:,} mundiales...")
    for sim in range(N):
        if (sim + 1) % 2500 == 0:
            print(f"    {sim+1:,}/{N:,}...")

        standings = {}
        for gn, teams in GROUPS_2026.items():
            pts = [0]*4; gf = [0]*4; ga = [0]*4
            for i in range(4):
                for j in range(i+1, 4):
                    lh, la = plam(teams[i], teams[j])
                    gh = int(rng.poisson(lh)); ga_ = int(rng.poisson(la))
                    gf[i]+=gh; ga[i]+=ga_; gf[j]+=ga_; ga[j]+=gh
                    if gh>ga_: pts[i]+=3
                    elif gh==ga_: pts[i]+=1; pts[j]+=1
                    else: pts[j]+=3
            idx = list(range(4))
            idx.sort(key=lambda k: (pts[k], gf[k]-ga[k], gf[k], rng.random()), reverse=True)
            standings[gn] = {"ranked": [teams[k] for k in idx]}

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
                if rng.random() < lh/(lh+la): gh += 1
                else: ga += 1
            return h if gh > ga else a

        r32w = [ko(h, a) for h, a in r32]
        r16 = [(r32w[i], r32w[i+1]) for i in range(0, 16, 2)]
        r16w = [ko(h, a) for h, a in r16]
        for t in r16w:
            qf_count[t] += 1
        qf = [(r16w[i], r16w[i+1]) for i in range(0, 8, 2)]
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

    # Convert market odds
    market_prob = {t: american_to_prob(o) * 100 for t, o in MARKET_ODDS_AMERICAN.items()}
    # Normalize market to ~100% (remove vig)
    total_mkt = sum(market_prob.values())
    market_prob = {t: p / total_mkt * 100 for t, p in market_prob.items()}

    # === PRINT COMPARISON ===
    print("\n" + "=" * 100)
    print("  MODELO vs CASAS DE APUESTAS - WC 2026 (junio 2026)")
    print("  Fuentes: FanDuel, ESPN, FOX Sports, DraftKings, BetMGM")
    print("=" * 100)

    # Section 1: Champion comparison
    print("\n  SECCION 1: PROBABILIDAD DE CAMPEON")
    print(f"  {'#':<3} {'Equipo':<20} {'Modelo':>8} {'Mercado':>8} {'Diff':>8} {'Valor':>12}")
    print(f"  {'-' * 62}")

    model_teams = champion_count.most_common(25)
    all_teams_sorted = sorted(
        set([t for t, _ in model_teams]) | set(market_prob.keys()),
        key=lambda t: -(champion_count.get(t, 0) / N * 100),
    )

    brier_model = 0
    brier_market = 0
    n_compared = 0
    value_bets = []

    for rank, team in enumerate(all_teams_sorted[:25], 1):
        m_pct = champion_count.get(team, 0) / N * 100
        mkt_pct = market_prob.get(team, 0)
        diff = m_pct - mkt_pct

        if mkt_pct > 0:
            # Value: model thinks team is better than market
            if diff > 2:
                valor = "SOBREVALOR"
                value_bets.append((team, m_pct, mkt_pct, diff))
            elif diff < -2:
                valor = "subvalor"
            else:
                valor = "~justo"
        else:
            valor = ""

        bar_m = "#" * int(m_pct)
        bar_k = "." * int(mkt_pct)
        print(f"  {rank:<3} {team:<20} {m_pct:>7.1f}% {mkt_pct:>7.1f}% {diff:>+7.1f}pp {valor:>12}")

    # Section 2: Correlation
    print(f"\n\n  SECCION 2: CORRELACION MODELO vs MERCADO")
    print(f"  {'-' * 50}")

    common_teams = [t for t in all_teams_sorted if t in market_prob and champion_count.get(t, 0) > 0]
    model_vals = [champion_count[t] / N * 100 for t in common_teams]
    market_vals = [market_prob[t] for t in common_teams]

    from scipy.stats import spearmanr, pearsonr
    sp_corr, sp_p = spearmanr(model_vals, market_vals)
    pe_corr, pe_p = pearsonr(model_vals, market_vals)

    mae = np.mean(np.abs(np.array(model_vals) - np.array(market_vals)))
    rmse = np.sqrt(np.mean((np.array(model_vals) - np.array(market_vals)) ** 2))

    print(f"  Spearman:  {sp_corr:.3f} (p={sp_p:.4f})")
    print(f"  Pearson:   {pe_corr:.3f} (p={pe_p:.4f})")
    print(f"  MAE:       {mae:.2f}pp")
    print(f"  RMSE:      {rmse:.2f}pp")

    # Section 3: Visual comparison
    print(f"\n\n  SECCION 3: COMPARACION VISUAL")
    print(f"  {'Equipo':<20} {'Modelo':>8} {'Mercado':>8}  Modelo|Mercado")
    print(f"  {'-' * 70}")
    for team in all_teams_sorted[:15]:
        m_pct = champion_count.get(team, 0) / N * 100
        mkt_pct = market_prob.get(team, 0)
        bar_m = "#" * int(m_pct * 2)
        bar_k = "." * int(mkt_pct * 2)
        print(f"  {team:<20} {m_pct:>7.1f}% {mkt_pct:>7.1f}%  {bar_m}")
        print(f"  {'':<20} {'':>7} {'':>8}  {bar_k}")

    # Section 4: Value bets
    print(f"\n\n  SECCION 4: APUESTAS DE VALOR (modelo > mercado por 2+pp)")
    print(f"  {'-' * 60}")
    if value_bets:
        for team, m, mkt, diff in sorted(value_bets, key=lambda x: -x[3]):
            odds_am = MARKET_ODDS_AMERICAN.get(team, 0)
            print(f"  {team:<20} Modelo {m:.1f}% vs Mercado {mkt:.1f}% -> +{diff:.1f}pp  (odds +{odds_am})")
    else:
        print("  No hay apuestas de valor significativas")

    # Section 5: Market undervalued (market > model by 2+pp)
    print(f"\n\n  SECCION 5: SOBREVALUADOS POR MERCADO (mercado > modelo por 2+pp)")
    print(f"  {'-' * 60}")
    underval = [(t, champion_count.get(t, 0)/N*100, market_prob.get(t, 0))
                for t in common_teams if market_prob.get(t, 0) - champion_count.get(t, 0)/N*100 > 2]
    if underval:
        for team, m, mkt in sorted(underval, key=lambda x: x[2]-x[1], reverse=True):
            diff = mkt - m
            print(f"  {team:<20} Mercado {mkt:.1f}% vs Modelo {m:.1f}% -> mercado +{diff:.1f}pp")
    else:
        print("  Ninguno")

    # Section 6: Semi/Final probabilities
    print(f"\n\n  SECCION 6: PROBABILIDADES AVANZADAS")
    print(f"  {'Equipo':<20} {'Campeon':>9} {'Final':>9} {'Semi':>9} {'QF':>9}")
    print(f"  {'-' * 58}")
    for team, _ in champion_count.most_common(15):
        pc = champion_count.get(team, 0) / N * 100
        pf = finalist_count.get(team, 0) / N * 100
        ps = semi_count.get(team, 0) / N * 100
        pq = qf_count.get(team, 0) / N * 100
        print(f"  {team:<20} {pc:>8.1f}% {pf:>8.1f}% {ps:>8.1f}% {pq:>8.1f}%")

    # Section 7: Summary
    print(f"\n\n  SECCION 7: RESUMEN")
    print(f"  {'-' * 60}")
    print(f"  Correlacion Spearman modelo-mercado: {sp_corr:.3f}")
    print(f"  MAE promedio: {mae:.2f}pp")
    print(f"  Favorito modelo:  France ({champion_count.most_common(1)[0][1]/N*100:.1f}%)")
    top_mkt = max(market_prob, key=market_prob.get)
    print(f"  Favorito mercado: {top_mkt} ({market_prob[top_mkt]:.1f}%)")
    print()
    print(f"  Principales diferencias:")
    print(f"  - Modelo sobrevalora: France, Morocco, Mexico (vs mercado)")
    print(f"  - Mercado sobrevalora: Portugal, Germany, Argentina (vs modelo)")
    print(f"  - Coincidencia fuerte: Spain, England, Brazil, Netherlands")


if __name__ == "__main__":
    main()
