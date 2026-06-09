"""Genera el cuadro completo de eliminatorias con equipos mas probables."""
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
    model = train_hybrid_model(df, cutoff)

    rng = np.random.default_rng(42)
    N = 10000

    def plam(h, a):
        sh = model.strength.get(h, 0.0)
        sa = model.strength.get(a, 0.0)
        d = sh - sa
        return math.exp(0.25 + d / model.scale), math.exp(0.25 - d / model.scale)

    # Track every knockout matchup and who wins
    r32_matchups = Counter()  # (slot, team1, team2)
    r32_winners = Counter()   # (slot, winner)
    r16_matchups = Counter()
    r16_winners = Counter()
    qf_matchups = Counter()
    qf_winners = Counter()
    sf_matchups = Counter()
    sf_winners = Counter()
    final_matchups = Counter()
    champion_count = Counter()

    # Track match-level results for most common matchups
    ko_match_results = {}  # (home, away) -> [(gh, ga), ...]

    print(f"Simulando {N:,} mundiales...", flush=True)

    for sim in range(N):
        if (sim + 1) % 2500 == 0:
            print(f"  {sim+1:,}/{N:,}...", flush=True)

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
            standings[gn] = [teams[k] for k in idx]

        thirds = []
        for g in sorted(standings):
            thirds.append(standings[g][2])
        tl = thirds[:8]

        winners = {g: standings[g][0] for g in standings}
        runners = {g: standings[g][1] for g in standings}

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

        r32_labels = [
            "R32-1", "R32-2", "R32-3", "R32-4",
            "R32-5", "R32-6", "R32-7", "R32-8",
            "R32-9", "R32-10", "R32-11", "R32-12",
            "R32-13", "R32-14", "R32-15", "R32-16",
        ]

        def ko(h, a):
            lh, la = plam(h, a)
            gh = int(rng.poisson(lh)); ga = int(rng.poisson(la))
            pen = ""
            if gh == ga:
                if rng.random() < lh/(lh+la): gh += 1
                else: ga += 1
                pen = " (pen)"
            key = tuple(sorted([h, a]))
            if key not in ko_match_results:
                ko_match_results[key] = {"w1": 0, "w2": 0, "draws_90": 0}
            if pen:
                ko_match_results[key]["draws_90"] += 1
            winner = h if gh > ga else a
            if winner == key[0]:
                ko_match_results[key]["w1"] += 1
            else:
                ko_match_results[key]["w2"] += 1
            return winner

        # R32
        for i, (h, a) in enumerate(r32):
            r32_matchups[(i, h, a)] += 1

        r32w = [ko(h, a) for h, a in r32]
        for i, w in enumerate(r32w):
            r32_winners[(i, w)] += 1

        # R16
        r16 = [(r32w[i], r32w[i+1]) for i in range(0, 16, 2)]
        r16_labels = ["R16-1", "R16-2", "R16-3", "R16-4",
                      "R16-5", "R16-6", "R16-7", "R16-8"]

        for i, (h, a) in enumerate(r16):
            r16_matchups[(i, h, a)] += 1

        r16w = [ko(h, a) for h, a in r16]
        for i, w in enumerate(r16w):
            r16_winners[(i, w)] += 1

        # QF
        qf = [(r16w[i], r16w[i+1]) for i in range(0, 8, 2)]
        for i, (h, a) in enumerate(qf):
            qf_matchups[(i, h, a)] += 1
        qfw = [ko(h, a) for h, a in qf]
        for i, w in enumerate(qfw):
            qf_winners[(i, w)] += 1

        # SF
        sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
        for i, (h, a) in enumerate(sf):
            sf_matchups[(i, h, a)] += 1
        sfw = [ko(h, a) for h, a in sf]
        for i, w in enumerate(sfw):
            sf_winners[(i, w)] += 1

        # Final
        final_matchups[tuple(sorted([sfw[0], sfw[1]]))] += 1
        champ = ko(sfw[0], sfw[1])
        champion_count[champ] += 1

    # === PRINT KNOCKOUT BRACKET ===
    def get_most_likely(counter_dict, slot, n=3):
        """Get top n teams for a given slot."""
        teams = [(t, c) for (s, t, *_), c in counter_dict.items() if s == slot]
        # Merge by team
        merged = Counter()
        for t, c in teams:
            merged[t] += c
        return merged.most_common(n)

    def get_most_likely_winners(counter_dict, slot, n=3):
        teams = [(t, c) for (s, t), c in counter_dict.items() if s == slot]
        merged = Counter()
        for t, c in teams:
            merged[t] += c
        return merged.most_common(n)

    def get_matchup_prob(team1, team2):
        """Get win probability for a specific matchup."""
        key = tuple(sorted([team1, team2]))
        if key not in ko_match_results:
            return 50, 50, 0
        d = ko_match_results[key]
        total = d["w1"] + d["w2"]
        if total == 0:
            return 50, 50, 0
        if key[0] == team1:
            return d["w1"]/total*100, d["w2"]/total*100, d["draws_90"]/total*100
        else:
            return d["w2"]/total*100, d["w1"]/total*100, d["draws_90"]/total*100

    print("\n" + "=" * 100)
    print("  CUADRO DE ELIMINATORIAS - WC 2026")
    print("  Equipos mas probables por posicion (10,000 simulaciones)")
    print("=" * 100)

    # R32 bracket structure
    r32_desc = [
        "1ro A vs 3ro mejor",    "2do C vs 2do D",
        "1ro B vs 3ro",          "2do E vs 2do F",
        "1ro G vs 3ro",          "2do I vs 2do J",
        "1ro H vs 3ro",          "2do K vs 2do L",
        "1ro C vs 3ro",          "2do A vs 2do B",
        "1ro D vs 3ro",          "2do G vs 2do H",
        "1ro I vs 3ro",          "1ro F vs 2do I",
        "1ro J vs 3ro",          "1ro L vs 2do K",
    ]

    print("\n  RONDA DE 32 (16 partidos)")
    print(f"  {'#':<6} {'Estructura':<25} {'Equipo mas probable':<45} {'Avanza':>20}")
    print(f"  {'-' * 100}")

    for slot in range(16):
        # Most likely matchup for this slot
        matchups_for_slot = [(h, a, c) for (s, h, a), c in r32_matchups.items() if s == slot]
        matchups_for_slot.sort(key=lambda x: -x[2])

        if matchups_for_slot:
            h, a, c = matchups_for_slot[0]
            pct = c / N * 100
            matchup_str = f"{h} vs {a} ({pct:.0f}%)"
        else:
            matchup_str = "?"

        # Most likely winner
        winners_for_slot = get_most_likely_winners(r32_winners, slot, 2)
        if winners_for_slot:
            w1, w1c = winners_for_slot[0]
            w_str = f"{w1} ({w1c/N*100:.0f}%)"
        else:
            w_str = "?"

        print(f"  {slot+1:<6} {r32_desc[slot]:<25} {matchup_str:<45} {w_str:>20}")

    # R16
    print(f"\n\n  OCTAVOS DE FINAL (8 partidos)")
    print(f"  {'#':<6} {'Partido mas probable':<50} {'Avanza':>20} {'Prob':>8}")
    print(f"  {'-' * 86}")

    r16_desc = ["R16-1 (R32-1 vs R32-2)", "R16-2 (R32-3 vs R32-4)",
                "R16-3 (R32-5 vs R32-6)", "R16-4 (R32-7 vs R32-8)",
                "R16-5 (R32-9 vs R32-10)", "R16-6 (R32-11 vs R32-12)",
                "R16-7 (R32-13 vs R32-14)", "R16-8 (R32-15 vs R32-16)"]

    for slot in range(8):
        matchups = [(h, a, c) for (s, h, a), c in r16_matchups.items() if s == slot]
        matchups.sort(key=lambda x: -x[2])

        if matchups:
            h, a, c = matchups[0]
            pct = c / N * 100
            w1p, w2p, draw_pct = get_matchup_prob(h, a)
            matchup_str = f"{h} vs {a} ({pct:.0f}%)"
        else:
            matchup_str = "?"
            w1p = w2p = 50

        winners = get_most_likely_winners(r16_winners, slot, 1)
        if winners:
            w, wc = winners[0]
            w_str = f"{w}"
            w_pct = wc / N * 100
        else:
            w_str = "?"
            w_pct = 0

        print(f"  {slot+1:<6} {matchup_str:<50} {w_str:>20} {w_pct:>7.1f}%")

    # QF
    print(f"\n\n  CUARTOS DE FINAL (4 partidos)")
    print(f"  {'#':<6} {'Partido mas probable':<50} {'Avanza':>20} {'Prob':>8}")
    print(f"  {'-' * 86}")

    for slot in range(4):
        matchups = [(h, a, c) for (s, h, a), c in qf_matchups.items() if s == slot]
        matchups.sort(key=lambda x: -x[2])

        if matchups:
            h, a, c = matchups[0]
            pct = c / N * 100
            matchup_str = f"{h} vs {a} ({pct:.0f}%)"
        else:
            matchup_str = "?"

        winners = get_most_likely_winners(qf_winners, slot, 1)
        if winners:
            w, wc = winners[0]
            w_pct = wc / N * 100
        else:
            w = "?"
            w_pct = 0

        print(f"  {slot+1:<6} {matchup_str:<50} {w:>20} {w_pct:>7.1f}%")

    # SF
    print(f"\n\n  SEMIFINALES (2 partidos)")
    print(f"  {'#':<6} {'Partido mas probable':<50} {'Avanza':>20} {'Prob':>8}")
    print(f"  {'-' * 86}")

    for slot in range(2):
        matchups = [(h, a, c) for (s, h, a), c in sf_matchups.items() if s == slot]
        matchups.sort(key=lambda x: -x[2])

        if matchups:
            h, a, c = matchups[0]
            pct = c / N * 100
            matchup_str = f"{h} vs {a} ({pct:.0f}%)"
        else:
            matchup_str = "?"

        winners = get_most_likely_winners(sf_winners, slot, 1)
        if winners:
            w, wc = winners[0]
            w_pct = wc / N * 100
        else:
            w = "?"
            w_pct = 0

        print(f"  {slot+1:<6} {matchup_str:<50} {w:>20} {w_pct:>7.1f}%")

    # Final
    print(f"\n\n  FINAL")
    print(f"  {'-' * 86}")
    for (t1, t2), c in final_matchups.most_common(5):
        pct = c / N * 100
        print(f"  {t1} vs {t2}: {pct:.1f}%")

    print(f"\n  CAMPEON")
    print(f"  {'-' * 40}")
    for team, c in champion_count.most_common(5):
        pct = c / N * 100
        bar = "#" * int(pct)
        print(f"  {team:<20} {pct:.1f}%  {bar}")


if __name__ == "__main__":
    main()
