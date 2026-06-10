"""Compare OLD model (14 features, pre-remediation) vs NEW model (clean, 4 L1 + DC).

Old model results extracted from git commit 6998d80 (pre-remediation).
New model runs fresh Monte Carlo simulation.
"""
import math
import numpy as np
import pandas as pd
from collections import Counter
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026, build_r32
from run_wc2026 import predict_lambdas_hybrid, sample_scoreline, sample_knockout_scoreline

# =====================================================================
# OLD MODEL DATA (from git commit 6998d80, pre-remediation)
# 14 features: elo, value, league, momentum, defense, champion,
#              market odds, age, coach, host, population, defense_l2,
#              diversity, composition, frontrunner curse
# Reported "Brier 0.0426" but with market odds leak
# =====================================================================
OLD_CHAMPION_PROBS = {
    "France": 19.6, "Spain": 17.4, "England": 15.8, "Argentina": 10.8,
    "Brazil": 9.7, "Morocco": 4.3, "Netherlands": 3.1, "Portugal": 2.8,
    "Japan": 2.5, "Mexico": 2.5, "Belgium": 2.5, "Germany": 1.8,
    "Norway": 1.3, "Switzerland": 0.9, "Turkey": 0.8, "Canada": 0.8,
    "Colombia": 0.7, "United States": 0.7, "Uruguay": 0.6, "Ecuador": 0.4,
}
OLD_FINALIST_PROBS = {
    "France": 31.2, "Spain": 30.8, "England": 23.8, "Brazil": 18.9,
    "Argentina": 18.4, "Morocco": 10.7, "Netherlands": 7.5, "Mexico": 7.5,
    "Belgium": 7.2, "Japan": 6.4, "Portugal": 5.6, "Norway": 4.2,
    "Germany": 4.0, "Switzerland": 3.5, "Canada": 3.2, "Turkey": 3.0,
    "Brazil": 18.9, "Uruguay": 2.6, "United States": 2.4, "Colombia": 2.3,
}
OLD_SEMI_PROBS = {
    "Spain": 45.9, "France": 43.3, "Brazil": 39.9, "England": 32.0,
    "Argentina": 27.2, "Morocco": 27.1, "Netherlands": 16.6, "Belgium": 16.2,
    "Japan": 15.1, "Mexico": 19.8, "Switzerland": 13.1, "Canada": 12.4,
    "Turkey": 11.0, "Uruguay": 10.9, "United States": 9.7, "Germany": 8.9,
    "Norway": 8.5, "Portugal": 8.7, "Ecuador": 5.8, "Colombia": 5.4,
}
OLD_GROUP_WINNERS = {
    "A": ("Mexico", 70.6), "B": ("Switzerland", 49.1), "C": ("Brazil", 57.4),
    "D": ("Turkey", 44.2), "E": ("Germany", 66.3), "F": ("Netherlands", 49.8),
    "G": ("Belgium", 74.6), "H": ("Spain", None), "I": ("France", None),
    "J": ("Argentina", None), "K": ("Portugal", None), "L": ("England", None),
}
OLD_MODAL_CHAMPION = "England"
OLD_MODEL_DESC = "14 features (incl. market odds, momentum, champion)"
OLD_RPS_CLAIMED = "0.1627 (in-sample, with market leak)"

NEW_MODEL_DESC = "4 L1 features (elo, value, league, defense) + Dixon-Coles"
NEW_RPS = "0.1992 (honest OOS, 14 tournaments, 610 matches)"


def sim_group_stage(model, groups, rng, N):
    """Run N group stage simulations, return stats."""
    group_pos = {g: {t: Counter() for t in teams} for g, teams in groups.items()}
    group_pts = {g: {t: [] for t in teams} for g, teams in groups.items()}
    group_gf = {g: {t: [] for t in teams} for g, teams in groups.items()}
    group_ga = {g: {t: [] for t in teams} for g, teams in groups.items()}

    for _ in range(N):
        for gn, teams in groups.items():
            pts = [0]*4; gf = [0]*4; ga = [0]*4
            for i in range(4):
                for j in range(i+1, 4):
                    lh, la = predict_lambdas_hybrid(model, teams[i], teams[j])
                    gh, ga_ = sample_scoreline(rng, lh, la)
                    gf[i] += gh; ga[i] += ga_; gf[j] += ga_; ga[j] += gh
                    if gh > ga_: pts[i] += 3
                    elif gh == ga_: pts[i] += 1; pts[j] += 1
                    else: pts[j] += 3
            idx = list(range(4))
            idx.sort(key=lambda k: (pts[k], gf[k]-ga[k], gf[k], rng.random()), reverse=True)
            for pos, k in enumerate(idx):
                group_pos[gn][teams[k]][pos+1] += 1
                group_pts[gn][teams[k]].append(pts[k])
                group_gf[gn][teams[k]].append(gf[k])
                group_ga[gn][teams[k]].append(ga[k])

    return group_pos, group_pts, group_gf, group_ga


def sim_tournament(model, groups, rng):
    """Simulate one full tournament, return (champion, finalist, semi_losers)."""
    # Group stage
    standings = {}
    for gn, teams in groups.items():
        pts = [0]*4; gf = [0]*4; ga = [0]*4
        for i in range(4):
            for j in range(i+1, 4):
                lh, la = predict_lambdas_hybrid(model, teams[i], teams[j])
                gh, ga_ = sample_scoreline(rng, lh, la)
                gf[i] += gh; ga[i] += ga_; gf[j] += ga_; ga[j] += gh
                if gh > ga_: pts[i] += 3
                elif gh == ga_: pts[i] += 1; pts[j] += 1
                else: pts[j] += 3
        idx = list(range(4))
        idx.sort(key=lambda k: (pts[k], gf[k]-ga[k], gf[k], rng.random()), reverse=True)
        standings[gn] = {
            "ranked": [teams[k] for k in idx],
            "pts": [pts[k] for k in idx],
            "gd": [gf[k]-ga[k] for k in idx],
            "gf": [gf[k] for k in idx],
        }

    winners = {g: s["ranked"][0] for g, s in standings.items()}
    runners = {g: s["ranked"][1] for g, s in standings.items()}
    thirds = []
    for g in sorted(standings):
        t = standings[g]["ranked"][2]
        thirds.append((g, t, standings[g]["pts"][2], standings[g]["gd"][2], standings[g]["gf"][2]))
    thirds.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = [t[1] for t in thirds[:8]]

    r32 = build_r32(winners, runners, best_thirds)

    # Knockout
    def ko_match(h, a):
        lh, la = predict_lambdas_hybrid(model, h, a)
        gh, ga_, _, _ = sample_knockout_scoreline(rng, lh, la)
        return h if gh > ga_ else a

    r32w = [ko_match(h, a) for h, a in r32]
    r16 = [(r32w[i], r32w[i+1]) for i in range(0, 16, 2)]
    r16w = [ko_match(h, a) for h, a in r16]
    qf = [(r16w[i], r16w[i+1]) for i in range(0, 8, 2)]
    qfw = [ko_match(h, a) for h, a in qf]
    sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
    sfw = [ko_match(h, a) for h, a in sf]
    sf_losers = [sf[i][0] if sfw[i] != sf[i][0] else sf[i][1] for i in range(2)]
    champ = ko_match(sfw[0], sfw[1])
    finalist = sfw[0] if champ != sfw[0] else sfw[1]

    return champ, finalist, sf_losers, set(qfw)


def main():
    N = 10000
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")

    print("Entrenando modelo actual (limpio)...")
    model = train_hybrid_model(df, cutoff)

    print(f"\nSimulando {N:,} torneos completos (Monte Carlo)...")
    rng = np.random.default_rng(42)

    champ_count = Counter()
    final_count = Counter()
    semi_count = Counter()
    qf_count = Counter()

    for i in range(N):
        if (i+1) % 2500 == 0:
            print(f"  {i+1:,}/{N:,}...")
        champ, finalist, sf_losers, qf_teams = sim_tournament(model, GROUPS_2026, rng)
        champ_count[champ] += 1
        final_count[champ] += 1
        final_count[finalist] += 1
        semi_count[champ] += 1
        semi_count[finalist] += 1
        for t in sf_losers:
            semi_count[t] += 1
        for t in qf_teams:
            qf_count[t] += 1

    # Also run group stage for group comparison
    print("Simulando fase de grupos para comparacion...")
    rng2 = np.random.default_rng(42)
    group_pos, group_pts, group_gf, group_ga = sim_group_stage(model, GROUPS_2026, rng2, N)

    # =====================================================================
    # BUILD COMPARISON REPORT
    # =====================================================================
    lines = []
    def out(s=""):
        lines.append(s)

    out("=" * 110)
    out("  COMPARACION: MODELO ANTES vs DESPUES DE REMEDIACION")
    out("  Simulaciones: {:,} por modelo | Fecha: 2026-06-10".format(N))
    out("=" * 110)

    out("\n" + "=" * 110)
    out("  SECCION 1: RESUMEN DE MODELOS")
    out("=" * 110)
    out(f"\n  {'':40} {'ANTES (overfitted)':>25} {'DESPUES (honesto)':>25}")
    out(f"  {'-'*92}")
    out(f"  {'Modelo:':<40} {'14 features + market':>25} {'4 L1 + Dixon-Coles':>25}")
    out(f"  {'RPS:':<40} {OLD_RPS_CLAIMED:>25} {NEW_RPS:>25}")
    out(f"  {'Dixon-Coles:':<40} {'No':>25} {'Si (rho={:.2f})'.format(model.rho):>25}")
    out(f"  {'Market odds:':<40} {'Si (32% L2)':>25} {'No (circular)':>25}")
    out(f"  {'Momentum:':<40} {'Si (14% L1)':>25} {'No (empeora RPS)':>25}")
    out(f"  {'Champion feature:':<40} {'Si (8% L1)':>25} {'No (empeora RPS)':>25}")
    out(f"  {'H2H:':<40} {'Si':>25} {'No (empeora RPS)':>25}")
    out(f"  {'Backtesting:':<40} {'Dudoso (leak)':>25} {'Walk-forward 14 torn.':>25}")

    # SECTION 2: CHAMPION PROBABILITIES SIDE BY SIDE
    out("\n\n" + "=" * 110)
    out("  SECCION 2: PROBABILIDAD DE CAMPEON (Monte Carlo {:,} sims)".format(N))
    out("=" * 110)

    new_champ = {t: c/N*100 for t, c in champ_count.items()}
    new_final = {t: c/N*100 for t, c in final_count.items()}
    new_semi = {t: c/N*100 for t, c in semi_count.items()}

    # Combine all teams
    all_teams_ranked = sorted(new_champ.keys(), key=lambda t: -new_champ.get(t, 0))

    out(f"\n  {'#':<4} {'Equipo':<22} {'ANTES':>8} {'DESPUES':>8} {'Diff':>8}   {'Cambio':>8}  {'Barra ANTES':>20} {'Barra DESPUES':>20}")
    out(f"  {'-'*108}")

    for i, t in enumerate(all_teams_ranked[:25], 1):
        old_p = OLD_CHAMPION_PROBS.get(t, 0.0)
        new_p = new_champ.get(t, 0.0)
        diff = new_p - old_p
        if old_p > 0:
            pct_change = diff / old_p * 100
            change_str = f"{pct_change:+.0f}%"
        else:
            change_str = "NEW" if new_p > 0 else ""

        bar_old = "#" * int(old_p * 2)
        bar_new = "#" * int(new_p * 2)
        arrow = ">>>" if diff > 2 else ("<<<" if diff < -2 else "  ~")
        out(f"  {i:<4} {t:<22} {old_p:>7.1f}% {new_p:>7.1f}% {diff:>+7.1f}pp {change_str:>8} {arrow} {bar_old:<20} {bar_new:<20}")

    # SECTION 3: BIGGEST MOVERS
    out("\n\n" + "=" * 110)
    out("  SECCION 3: MAYORES CAMBIOS (Campeon %)")
    out("=" * 110)

    movers = []
    all_t = set(list(OLD_CHAMPION_PROBS.keys()) + list(new_champ.keys()))
    for t in all_t:
        old_p = OLD_CHAMPION_PROBS.get(t, 0.0)
        new_p = new_champ.get(t, 0.0)
        movers.append((t, old_p, new_p, new_p - old_p))

    out("\n  SUBIERON:")
    up = sorted([m for m in movers if m[3] > 0.1], key=lambda x: -x[3])
    for t, old_p, new_p, diff in up[:10]:
        bar = "+" * int(abs(diff) * 3)
        out(f"    {t:<22} {old_p:>6.1f}% -> {new_p:>6.1f}%  ({diff:+.1f}pp) {bar}")

    out("\n  BAJARON:")
    down = sorted([m for m in movers if m[3] < -0.1], key=lambda x: x[3])
    for t, old_p, new_p, diff in down[:10]:
        bar = "-" * int(abs(diff) * 3)
        out(f"    {t:<22} {old_p:>6.1f}% -> {new_p:>6.1f}%  ({diff:+.1f}pp) {bar}")

    # SECTION 4: FINALIST COMPARISON
    out("\n\n" + "=" * 110)
    out("  SECCION 4: PROBABILIDAD DE FINALISTA")
    out("=" * 110)

    out(f"\n  {'#':<4} {'Equipo':<22} {'ANTES':>8} {'DESPUES':>8} {'Diff':>8}")
    out(f"  {'-'*55}")
    for i, t in enumerate(all_teams_ranked[:20], 1):
        old_p = OLD_FINALIST_PROBS.get(t, 0.0)
        new_p = new_final.get(t, 0.0)
        diff = new_p - old_p
        out(f"  {i:<4} {t:<22} {old_p:>7.1f}% {new_p:>7.1f}% {diff:>+7.1f}pp")

    # SECTION 5: SEMIFINALIST COMPARISON
    out("\n\n" + "=" * 110)
    out("  SECCION 5: PROBABILIDAD DE SEMIFINALISTA")
    out("=" * 110)

    out(f"\n  {'#':<4} {'Equipo':<22} {'ANTES':>8} {'DESPUES':>8} {'Diff':>8}")
    out(f"  {'-'*55}")
    for i, t in enumerate(all_teams_ranked[:20], 1):
        old_p = OLD_SEMI_PROBS.get(t, 0.0)
        new_p = new_semi.get(t, 0.0)
        diff = new_p - old_p
        out(f"  {i:<4} {t:<22} {old_p:>7.1f}% {new_p:>7.1f}% {diff:>+7.1f}pp")

    # SECTION 6: GROUP WINNERS COMPARISON
    out("\n\n" + "=" * 110)
    out("  SECCION 6: GANADORES DE GRUPO")
    out("=" * 110)

    out(f"\n  {'Grupo':<7} {'1ro ANTES':<22} {'1ro DESPUES':<22} {'Cambio?':>10}")
    out(f"  {'-'*65}")
    for g in sorted(GROUPS_2026.keys()):
        old_winner = OLD_GROUP_WINNERS.get(g, ("?", 0))[0]
        # New winner
        teams = GROUPS_2026[g]
        best_t = max(teams, key=lambda t: (group_pos[g][t].get(1, 0) + group_pos[g][t].get(2, 0)))
        new_1st = max(teams, key=lambda t: group_pos[g][t].get(1, 0))
        new_p1 = group_pos[g][new_1st].get(1, 0) / N * 100
        old_p1 = OLD_GROUP_WINNERS.get(g, ("?", 0))[1]
        old_p_str = f"({old_p1:.0f}%)" if old_p1 else ""
        changed = "CAMBIO" if old_winner != new_1st else "="
        out(f"  {g:<7} {old_winner+' '+old_p_str:<22} {new_1st+f' ({new_p1:.0f}%)':<22} {changed:>10}")

    # SECTION 7: STRENGTH RANKING COMPARISON
    out("\n\n" + "=" * 110)
    out("  SECCION 7: RANKING DE FUERZA DEL MODELO NUEVO")
    out("=" * 110)

    wc_teams = set()
    for teams in GROUPS_2026.values():
        wc_teams.update(teams)
    strengths = sorted([(t, model.strength.get(t, 0.0)) for t in wc_teams], key=lambda x: -x[1])

    out(f"\n  {'#':<4} {'Equipo':<22} {'Fuerza':>8} {'Campeon%':>9}")
    out(f"  {'-'*47}")
    for i, (t, s) in enumerate(strengths[:25], 1):
        cp = new_champ.get(t, 0.0)
        out(f"  {i:<4} {t:<22} {s:>7.3f} {cp:>8.1f}%")

    # SECTION 8: NARRATIVE SUMMARY
    out("\n\n" + "=" * 110)
    out("  SECCION 8: INTERPRETACION")
    out("=" * 110)

    new_top1 = all_teams_ranked[0]
    new_top1_p = new_champ[new_top1]
    old_top1 = "France"
    old_top1_p = 19.6

    out(f"""
  El modelo ANTES favorecia a {old_top1} ({old_top1_p:.1f}%) como campeon.
  El modelo DESPUES favorece a {new_top1} ({new_top1_p:.1f}%).

  Cambios principales tras la remediacion:

  1. MARKET ODDS ELIMINADAS: El modelo viejo usaba odds de casas de apuestas
     como feature (32% del Layer 2). Esto era circular: comparar el modelo
     contra el mercado cuando el modelo USA el mercado no tiene sentido.

  2. MOMENTUM/CHAMPION ELIMINADOS: Features narrativas que agregaban ruido.
     La ablation mostro que cada una empeoraba el RPS out-of-sample.

  3. DIXON-COLES AGREGADO: Correccion de correlacion en marcadores bajos
     (0-0, 1-0, 0-1, 1-1). Mejora calibracion de empates.

  4. RPS HONESTO: El modelo viejo reportaba RPS 0.1627 pero con leak de
     market odds. El nuevo reporta 0.1992 sobre 14 torneos walk-forward.
     Es un numero peor, pero REAL. El viejo era una ilusion.

  5. CONCENTRACION DEL PODER: Al remover features ruidosas, el modelo
     se concentra mas en las senales reales (Elo + valor de mercado).
     Esto puede cambiar la distribucion de probabilidades.""")

    # SECTION 9: MODEL DETAILS
    out("\n\n" + "=" * 110)
    out("  SECCION 9: DETALLE TECNICO")
    out("=" * 110)
    out(f"\n  Modelo actual:")
    out(f"    Scale: {model.scale:.2f}")
    out(f"    Home advantage: {model.home_adv:.2f}")
    out(f"    Dixon-Coles rho: {model.rho:.2f}")
    out(f"    L1 weights: {', '.join(f'{k}={v:.2f}' for k,v in model.weights.items() if k.startswith('L1_'))}")
    out(f"    L2 weights: {', '.join(f'{k}={v:.2f}' for k,v in model.weights.items() if not k.startswith('L1_'))}")
    out(f"\n  RPS out-of-sample: {NEW_RPS}")
    out(f"  Backtest: 14 torneos (WC 2010/14/18/22, Euro 2012/16/20/24, Copa Am 2011/15/16/19/21/24)")
    out(f"\n  Modelo anterior (eliminado):")
    out(f"    Features: {OLD_MODEL_DESC}")
    out(f"    RPS reportado: {OLD_RPS_CLAIMED}")

    txt = "\n".join(lines)
    outfile = "COMPARACION_ANTES_DESPUES.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nGuardado: {outfile} ({len(lines)} lineas)")

    # Also print to console
    print("\n" + txt)


if __name__ == "__main__":
    main()
