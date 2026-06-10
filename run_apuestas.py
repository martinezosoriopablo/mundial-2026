"""Sistema de apuestas combinadas (parlays) para Mundial 2026.

Genera combinaciones de 2-3 partidos con alta probabilidad,
usando apuestas simples (1/X/2) y doble chance (1X/X2/12).
Partidos dentro de ventanas de 2 dias para combinadas realizables.
"""
import math
import itertools
import numpy as np
import pandas as pd
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026
from run_wc2026 import predict_lambdas_hybrid

# ======================================================================
# WC 2026 Group Stage Schedule (FIFA official structure)
# 72 matches: June 11 - June 28, 2026
# ======================================================================

def build_schedule():
    """Build realistic WC2026 group stage schedule.

    3 matchdays per group, spread across 18 days.
    Each group's matches are spaced ~5 days apart.
    4 matches per day.
    """
    # Group match pairings: (team_idx_home, team_idx_away)
    # MD1: 0v1, 2v3  |  MD2: 0v2, 1v3  |  MD3: 0v3, 1v2
    md_pairings = [
        [(0, 1), (2, 3)],  # MD1
        [(0, 2), (1, 3)],  # MD2
        [(0, 3), (1, 2)],  # MD3
    ]

    # Assign matchdays to dates (FIFA-style: 4 matches/day)
    # MD1: June 11-15 (groups A-L spread across 5 days, 2 groups/day, 4 matches/day)
    # MD2: June 17-21
    # MD3: June 24-28

    md1_dates = {
        "A": "2026-06-11", "B": "2026-06-11",
        "C": "2026-06-12", "D": "2026-06-12",
        "E": "2026-06-13", "F": "2026-06-13",
        "G": "2026-06-14", "H": "2026-06-14",
        "I": "2026-06-15", "J": "2026-06-15",
        "K": "2026-06-16", "L": "2026-06-16",
    }
    md2_dates = {
        "A": "2026-06-18", "B": "2026-06-18",
        "C": "2026-06-19", "D": "2026-06-19",
        "E": "2026-06-20", "F": "2026-06-20",
        "G": "2026-06-21", "H": "2026-06-21",
        "I": "2026-06-22", "J": "2026-06-22",
        "K": "2026-06-23", "L": "2026-06-23",
    }
    md3_dates = {
        "A": "2026-06-25", "B": "2026-06-25",
        "C": "2026-06-26", "D": "2026-06-26",
        "E": "2026-06-27", "F": "2026-06-27",
        "G": "2026-06-28", "H": "2026-06-28",
        "I": "2026-06-25", "J": "2026-06-26",
        "K": "2026-06-27", "L": "2026-06-28",
    }

    all_dates = [md1_dates, md2_dates, md3_dates]

    schedule = []
    for gname in sorted(GROUPS_2026.keys()):
        teams = GROUPS_2026[gname]
        for md_idx, pairings in enumerate(md_pairings):
            date = all_dates[md_idx][gname]
            for i, j in pairings:
                schedule.append({
                    "date": date,
                    "group": gname,
                    "matchday": md_idx + 1,
                    "home": teams[i],
                    "away": teams[j],
                })
    schedule.sort(key=lambda x: (x["date"], x["group"]))
    return schedule


def compute_bet_types(model, home, away):
    """Compute all bet type probabilities and fair odds."""
    p = model.predict_match(home, away, neutral=True)
    p1, px, p2 = p[0], p[1], p[2]

    bets = {
        "1": p1,
        "X": px,
        "2": p2,
        "1X": p1 + px,
        "X2": px + p2,
        "12": p1 + p2,
    }
    return bets


def fair_to_bookmaker_odds(prob, margin=0.07):
    """Convert fair probability to approximate bookmaker decimal odds."""
    if prob <= 0:
        return 999.0
    fair_odds = 1.0 / prob
    # Bookmaker takes margin, so odds are lower than fair
    bk_odds = fair_odds * (1 - margin)
    return max(bk_odds, 1.01)


def find_parlays(all_bets, schedule, max_day_gap=2, min_prob=0.75,
                 parlay_sizes=[2, 3]):
    """Find optimal parlays within date windows.

    Args:
        all_bets: list of (match_id, date, group, home, away, bet_type, prob, odds)
        schedule: list of match dicts
        max_day_gap: max days between first and last match in parlay
        min_prob: minimum probability for individual bets
        parlay_sizes: list of parlay sizes to consider (2, 3)

    Returns:
        list of parlays sorted by combined probability
    """
    # Filter to safe bets only
    safe_bets = [b for b in all_bets if b[6] >= min_prob]

    # Group by date
    date_to_bets = {}
    for b in safe_bets:
        d = b[1]
        if d not in date_to_bets:
            date_to_bets[d] = []
        date_to_bets[d].append(b)

    dates_sorted = sorted(date_to_bets.keys())

    parlays = []

    for size in parlay_sizes:
        # Find all date windows of max_day_gap
        for i, d1 in enumerate(dates_sorted):
            window_dates = [d1]
            for j in range(i + 1, len(dates_sorted)):
                d2 = dates_sorted[j]
                day_diff = (pd.Timestamp(d2) - pd.Timestamp(d1)).days
                if day_diff <= max_day_gap:
                    window_dates.append(d2)
                else:
                    break

            # Collect all bets in this window
            window_bets = []
            for d in window_dates:
                window_bets.extend(date_to_bets.get(d, []))

            # Only keep best bet per match (avoid betting same match twice)
            best_per_match = {}
            for b in window_bets:
                mid = (b[3], b[4])  # (home, away) as match ID
                if mid not in best_per_match or b[6] > best_per_match[mid][6]:
                    best_per_match[mid] = b

            unique_bets = list(best_per_match.values())

            if len(unique_bets) < size:
                continue

            # Find top combinations
            for combo in itertools.combinations(unique_bets, size):
                # Ensure no duplicate matches
                match_ids = set((b[3], b[4]) for b in combo)
                if len(match_ids) < size:
                    continue

                combined_prob = 1.0
                combined_odds = 1.0
                for b in combo:
                    combined_prob *= b[6]
                    combined_odds *= b[7]

                if combined_prob < 0.30:  # Skip very low combined probs
                    continue

                parlays.append({
                    "bets": combo,
                    "size": size,
                    "combined_prob": combined_prob,
                    "combined_odds": combined_odds,
                    "dates": sorted(set(b[1] for b in combo)),
                    "potential_return_per_unit": combined_odds,
                })

    # Sort by combined probability (safest first)
    parlays.sort(key=lambda x: -x["combined_prob"])
    return parlays


def main():
    print("Cargando modelo...")
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    model = train_hybrid_model(df, cutoff)

    schedule = build_schedule()

    # Compute all bets for all matches
    all_bets = []
    match_probs = {}

    for m in schedule:
        h, a = m["home"], m["away"]
        bets = compute_bet_types(model, h, a)
        match_probs[(h, a)] = bets

        for btype, prob in bets.items():
            odds = fair_to_bookmaker_odds(prob)
            all_bets.append((
                f"{m['group']}", m["date"], m["group"],
                h, a, btype, prob, odds
            ))

    # ======================================================================
    # REPORT
    # ======================================================================
    lines = []
    def out(s=""):
        lines.append(s)

    out("=" * 110)
    out("  SISTEMA DE APUESTAS COMBINADAS - MUNDIAL FIFA 2026")
    out("  Modelo: Hybrid + Dixon-Coles + H2H | Scale={:.2f} | Upset sigma=0.15".format(model.scale))
    out("  Estrategia: Parlays de 2-3 partidos en ventanas de 2 dias")
    out("=" * 110)

    # ======================================================================
    # SECTION 1: ALL MATCHES WITH BET PROBABILITIES
    # ======================================================================
    out("\n")
    out("=" * 110)
    out("  SECCION 1: CALENDARIO COMPLETO CON PROBABILIDADES DE APUESTA")
    out("=" * 110)

    current_date = None
    out(f"\n  {'Fecha':<12} {'Gr':<3} {'Partido':<42} {'1':>7} {'X':>7} {'2':>7} {'1X':>7} {'X2':>7} {'12':>7} {'Mejor apuesta':<16}")
    out(f"  {'-'*115}")

    for m in schedule:
        h, a = m["home"], m["away"]
        b = match_probs[(h, a)]
        # Find best bet
        best_type = max(b.items(), key=lambda x: x[1])
        best_label = f"{best_type[0]} ({best_type[1]:.0%})"

        if m["date"] != current_date:
            if current_date is not None:
                out("")
            current_date = m["date"]

        label = f"{h} vs {a}"
        out(f"  {m['date']:<12} {m['group']:<3} {label:<42} {b['1']:>6.1%} {b['X']:>6.1%} {b['2']:>6.1%} {b['1X']:>6.1%} {b['X2']:>6.1%} {b['12']:>6.1%} {best_label:<16}")

    # ======================================================================
    # SECTION 2: TOP SINGLE BETS (safest individual bets)
    # ======================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 2: TOP 30 APUESTAS INDIVIDUALES MAS SEGURAS")
    out("=" * 110)
    out(f"\n  {'#':<4} {'Fecha':<12} {'Gr':<3} {'Partido':<40} {'Tipo':<6} {'Prob':>7} {'Cuota':>7} {'Ganancia/100':>13}")
    out(f"  {'-'*95}")

    all_bets_sorted = sorted(all_bets, key=lambda x: -x[6])
    shown = set()
    count = 0
    for b in all_bets_sorted:
        if count >= 30:
            break
        # Skip if we already showed a bet for this match with higher prob
        mid = (b[3], b[4])
        if mid in shown:
            continue
        shown.add(mid)
        count += 1
        gain = (b[7] - 1) * 100
        out(f"  {count:<4} {b[1]:<12} {b[2]:<3} {b[3]+' vs '+b[4]:<40} {b[5]:<6} {b[6]:>6.1%} {b[7]:>6.2f} {gain:>+12.1f}")

    # ======================================================================
    # SECTION 3: COMBINED BETS (PARLAYS) - 2 matches
    # ======================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 3: APUESTAS DOBLES (2 partidos, ventana 2 dias)")
    out("  Probabilidad minima individual: 80% | Ordenadas por probabilidad combinada")
    out("=" * 110)

    parlays = find_parlays(all_bets, schedule, max_day_gap=2,
                           min_prob=0.80, parlay_sizes=[2])

    out(f"\n  {'#':<4} {'Prob comb':>10} {'Cuota':>7} {'Gan/100':>8}  {'Fechas':<25} Apuestas")
    out(f"  {'-'*110}")

    for i, p in enumerate(parlays[:30], 1):
        dates_str = " | ".join(p["dates"])
        bets_str = " + ".join(
            f"{b[3]} vs {b[4]} [{b[5]}]({b[6]:.0%})"
            for b in p["bets"]
        )
        gain = (p["combined_odds"] - 1) * 100
        out(f"  {i:<4} {p['combined_prob']:>9.1%} {p['combined_odds']:>6.2f} {gain:>+7.1f}  {dates_str:<25} {bets_str}")

    # ======================================================================
    # SECTION 4: COMBINED BETS (PARLAYS) - 3 matches
    # ======================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 4: APUESTAS TRIPLES (3 partidos, ventana 2 dias)")
    out("  Probabilidad minima individual: 80% | Ordenadas por probabilidad combinada")
    out("=" * 110)

    parlays3 = find_parlays(all_bets, schedule, max_day_gap=2,
                            min_prob=0.80, parlay_sizes=[3])

    out(f"\n  {'#':<4} {'Prob comb':>10} {'Cuota':>7} {'Gan/100':>8}  {'Fechas':<25} Apuestas")
    out(f"  {'-'*110}")

    for i, p in enumerate(parlays3[:30], 1):
        dates_str = " | ".join(p["dates"])
        bets_str = " + ".join(
            f"{b[3]}vs{b[4]}[{b[5]}]({b[6]:.0%})"
            for b in p["bets"]
        )
        gain = (p["combined_odds"] - 1) * 100
        out(f"  {i:<4} {p['combined_prob']:>9.1%} {p['combined_odds']:>6.2f} {gain:>+7.1f}  {dates_str:<25} {bets_str}")

    # ======================================================================
    # SECTION 5: BEST VALUE PARLAYS (highest return for >50% probability)
    # ======================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 5: MEJORES COMBINADAS POR VALOR (prob >50%, max retorno)")
    out("  Incluye apuestas con prob individual >70%")
    out("=" * 110)

    value_parlays = find_parlays(all_bets, schedule, max_day_gap=2,
                                 min_prob=0.70, parlay_sizes=[2, 3])
    # Filter to >50% combined and sort by odds (value)
    value_parlays = [p for p in value_parlays if p["combined_prob"] >= 0.50]
    value_parlays.sort(key=lambda x: -x["combined_odds"])

    out(f"\n  {'#':<4} {'Tam':>4} {'Prob comb':>10} {'Cuota':>7} {'Gan/100':>8}  Apuestas")
    out(f"  {'-'*110}")

    for i, p in enumerate(value_parlays[:30], 1):
        bets_str = " + ".join(
            f"{b[3]}vs{b[4]}[{b[5]}]({b[6]:.0%})"
            for b in p["bets"]
        )
        gain = (p["combined_odds"] - 1) * 100
        out(f"  {i:<4} {p['size']:>4} {p['combined_prob']:>9.1%} {p['combined_odds']:>6.2f} {gain:>+7.1f}  {bets_str}")

    # ======================================================================
    # SECTION 6: DAY-BY-DAY BETTING CALENDAR
    # ======================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 6: CALENDARIO DE APUESTAS DIA A DIA")
    out("  Para cada dia, la mejor combinada disponible")
    out("=" * 110)

    # Group matches by date
    by_date = {}
    for m in schedule:
        d = m["date"]
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(m)

    for date in sorted(by_date.keys()):
        matches_today = by_date[date]
        out(f"\n  {'='*30} {date} {'='*30}")

        # Best bet per match today
        day_bets = []
        for m in matches_today:
            h, a = m["home"], m["away"]
            b = match_probs[(h, a)]
            best_type = max(b.items(), key=lambda x: x[1])
            day_bets.append((m, best_type[0], best_type[1]))
            odds = fair_to_bookmaker_odds(best_type[1])
            label = f"{h} vs {a}"
            out(f"    {label:<42} -> {best_type[0]:<4} {best_type[1]:>6.1%} (cuota {odds:.2f})")

        # Best double from today's matches
        if len(day_bets) >= 2:
            day_bets.sort(key=lambda x: -x[2])
            top2 = day_bets[:2]
            comb_prob = top2[0][2] * top2[1][2]
            comb_odds = fair_to_bookmaker_odds(top2[0][2]) * fair_to_bookmaker_odds(top2[1][2])
            gain = (comb_odds - 1) * 100
            out(f"\n    >> DOBLE DEL DIA: {top2[0][0]['home']} vs {top2[0][0]['away']} [{top2[0][1]}]")
            out(f"                   + {top2[1][0]['home']} vs {top2[1][0]['away']} [{top2[1][1]}]")
            out(f"       Prob combinada: {comb_prob:.1%} | Cuota: {comb_odds:.2f} | Ganancia/100: {gain:+.1f}")

    # ======================================================================
    # SECTION 7: STRATEGIC SUMMARY
    # ======================================================================
    out("\n\n")
    out("=" * 110)
    out("  SECCION 7: RESUMEN ESTRATEGICO")
    out("=" * 110)

    out(f"\n  Partidos totales fase de grupos: 72")
    out(f"  Apuestas individuales >80%: {len([b for b in all_bets if b[6] > 0.80])}")
    out(f"  Apuestas individuales >85%: {len([b for b in all_bets if b[6] > 0.85])}")
    out(f"  Apuestas individuales >90%: {len([b for b in all_bets if b[6] > 0.90])}")

    out(f"\n  --- ESTRATEGIA RECOMENDADA ---")
    out(f"  1. CONSERVADORA (prob >80%): Dobles con apuestas 1X/X2 de favoritos claros")
    out(f"     Retorno esperado: +10-20% por combinada")
    out(f"     Tasa de acierto estimada: ~70%")
    out(f"")
    out(f"  2. MODERADA (prob >70%): Triples mezclando 1X y resultados directos")
    out(f"     Retorno esperado: +30-60% por combinada")
    out(f"     Tasa de acierto estimada: ~50%")
    out(f"")
    out(f"  3. AGRESIVA (prob >60%): Triples con resultados directos de favoritos")
    out(f"     Retorno esperado: +80-150% por combinada")
    out(f"     Tasa de acierto estimada: ~30%")

    out(f"\n  --- PARTIDOS TRAMPA (evitar en combinadas) ---")
    # Matches where all outcomes are close to 33%
    traps = []
    for m in schedule:
        h, a = m["home"], m["away"]
        b = match_probs[(h, a)]
        max_prob = max(b["1"], b["X"], b["2"])
        if max_prob < 0.42:
            traps.append((m, max_prob))
    traps.sort(key=lambda x: x[1])
    for m, mp in traps[:10]:
        b = match_probs[(m["home"], m["away"])]
        out(f"    {m['date']} Gr.{m['group']} {m['home']} vs {m['away']}: "
            f"1={b['1']:.0%} X={b['X']:.0%} 2={b['2']:.0%} (max {mp:.0%})")

    # Write
    txt = "\n".join(lines)
    outfile = "APUESTAS_WC2026.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nGuardado: {outfile} ({len(lines)} lineas)")

    # Print highlights
    print("\n=== HIGHLIGHTS ===")
    if parlays:
        print(f"Mejor doble segura: {parlays[0]['combined_prob']:.1%} prob, cuota {parlays[0]['combined_odds']:.2f}")
        for b in parlays[0]["bets"]:
            print(f"  - {b[3]} vs {b[4]} [{b[5]}] ({b[6]:.0%})")
    if parlays3:
        print(f"\nMejor triple segura: {parlays3[0]['combined_prob']:.1%} prob, cuota {parlays3[0]['combined_odds']:.2f}")
        for b in parlays3[0]["bets"]:
            print(f"  - {b[3]} vs {b[4]} [{b[5]}] ({b[6]:.0%})")
    if value_parlays:
        print(f"\nMejor valor (>50% prob): {value_parlays[0]['combined_prob']:.1%} prob, cuota {value_parlays[0]['combined_odds']:.2f}")
        for b in value_parlays[0]["bets"]:
            print(f"  - {b[3]} vs {b[4]} [{b[5]}] ({b[6]:.0%})")


if __name__ == "__main__":
    main()
