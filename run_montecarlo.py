"""Monte Carlo simulation of World Cup 2026 - champion probabilities.

Compares two goal-sampling methods:
  1. Poisson (classic): variance = mean
  2. Negative Binomial: variance > mean (more upsets, more realistic)
"""

import math
import numpy as np
import pandas as pd
from src.data_loader import load_results
from src.strength_elo import train_elo_model
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026, build_r32
from run_wc2026 import (
    predict_lambdas_elo,
    predict_lambdas_hybrid,
    sample_scoreline,
    sample_knockout_scoreline,
)


ROTATION_FACTOR = 0.75  # strength reduction when team already qualified plays MD3 subs

# Matchday pairings for a 4-team group (indices 0,1,2,3)
_MD_PAIRINGS = [
    [(0, 1), (2, 3)],  # MD1
    [(0, 2), (1, 3)],  # MD2
    [(0, 3), (1, 2)],  # MD3
]


def _apply_upset_noise(lh, la, rng, upset_sigma):
    """Shift lambda ratio by random noise, preserving total expected goals."""
    if upset_sigma > 0:
        total = lh + la
        noise = rng.normal(0, upset_sigma)
        lh_new = lh * math.exp(noise)
        la_new = la * math.exp(-noise)
        # Rescale to preserve total expected goals (counteracts Jensen's inequality)
        scale = total / (lh_new + la_new)
        lh = lh_new * scale
        la = la_new * scale
    return lh, la


def sim_group_stage_fast(model, get_lambdas, groups, rng, method="poisson", r=8.0,
                         upset_sigma=0.0):
    """Simulate group stage by matchday, return (standings dict, best 3rd place teams).

    After MD2, teams with 6 pts (won both) are considered qualified and
    play MD3 with reduced strength (rotation/subs), reflecting real
    tournament behavior where coaches rest starters.
    """
    standings = {}

    for gname, teams in groups.items():
        pts = [0] * 4
        gf = [0] * 4
        ga = [0] * 4

        # Play MD1 and MD2 normally
        for md_idx in range(2):
            for i, j in _MD_PAIRINGS[md_idx]:
                lh, la = get_lambdas(model, teams[i], teams[j], neutral=True)
                lh, la = _apply_upset_noise(lh, la, rng, upset_sigma)
                gh, ga_ = sample_scoreline(rng, lh, la, method, r)
                gf[i] += gh; ga[i] += ga_
                gf[j] += ga_; ga[j] += gh
                if gh > ga_:
                    pts[i] += 3
                elif gh == ga_:
                    pts[i] += 1; pts[j] += 1
                else:
                    pts[j] += 3

        # Detect qualified teams (6 pts after MD2 = won both games)
        qualified = {k for k in range(4) if pts[k] >= 6}

        # Play MD3 with rotation adjustment for qualified teams
        for i, j in _MD_PAIRINGS[2]:
            lh, la = get_lambdas(model, teams[i], teams[j], neutral=True)
            lh, la = _apply_upset_noise(lh, la, rng, upset_sigma)
            # Reduce lambdas for teams rotating subs
            if i in qualified:
                lh *= ROTATION_FACTOR
            if j in qualified:
                la *= ROTATION_FACTOR
            gh, ga_ = sample_scoreline(rng, lh, la, method, r)
            gf[i] += gh; ga[i] += ga_
            gf[j] += ga_; ga[j] += gh
            if gh > ga_:
                pts[i] += 3
            elif gh == ga_:
                pts[i] += 1; pts[j] += 1
            else:
                pts[j] += 3

        indices = list(range(4))
        indices.sort(key=lambda k: (pts[k], gf[k] - ga[k], gf[k], rng.random()), reverse=True)
        standings[gname] = {
            "ranked": [teams[k] for k in indices],
            "pts": [pts[k] for k in indices],
            "gd": [gf[k] - ga[k] for k in indices],
            "gf": [gf[k] for k in indices],
        }

    thirds = []
    for g in sorted(standings):
        s = standings[g]
        thirds.append((g, s["ranked"][2], s["pts"][2], s["gd"][2], s["gf"][2]))
    thirds.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = [t[1] for t in thirds[:8]]

    return standings, best_thirds


def sim_knockout_match(model, get_lambdas, home, away, rng, method="poisson", r=8.0,
                       upset_sigma=1.10):
    """Simulate a knockout match with tournament upset factor."""
    lh, la = get_lambdas(model, home, away, neutral=True)
    lh, la = _apply_upset_noise(lh, la, rng, upset_sigma)
    gh, ga, _, _ = sample_knockout_scoreline(rng, lh, la, method, r)
    return home if gh > ga else away


def sim_tournament(model, get_lambdas, groups, rng, method="poisson", r=8.0,
                   upset_sigma=1.10):
    """Simulate full tournament, return (champion, finalist, semifinalists)."""
    standings, best_thirds = sim_group_stage_fast(model, get_lambdas, groups, rng, method, r, upset_sigma)

    winners = {g: standings[g]["ranked"][0] for g in standings}
    runners = {g: standings[g]["ranked"][1] for g in standings}

    tl = best_thirds
    r32 = build_r32(winners, runners, tl)

    r32w = [sim_knockout_match(model, get_lambdas, h, a, rng, method, r, upset_sigma) for h, a in r32]
    r16 = [(r32w[i], r32w[i + 1]) for i in range(0, 16, 2)]
    r16w = [sim_knockout_match(model, get_lambdas, h, a, rng, method, r, upset_sigma) for h, a in r16]
    qf = [(r16w[i], r16w[i + 1]) for i in range(0, 8, 2)]
    qfw = [sim_knockout_match(model, get_lambdas, h, a, rng, method, r, upset_sigma) for h, a in qf]
    sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
    sfw = [sim_knockout_match(model, get_lambdas, h, a, rng, method, r, upset_sigma) for h, a in sf]
    champion = sim_knockout_match(model, get_lambdas, sfw[0], sfw[1], rng, method, r, upset_sigma)
    finalist = sfw[1] if champion == sfw[0] else sfw[0]

    sf_losers = []
    for i, (h, a) in enumerate(sf):
        sf_losers.append(a if sfw[i] == h else h)

    return champion, finalist, sf_losers, qfw


def run_montecarlo(model, get_lambdas, model_name, n_sims=10000, seed=42,
                   method="poisson", r=8.0, upset_sigma=1.10):
    """Run Monte Carlo tournament simulation."""
    rng = np.random.default_rng(seed)

    champion_count = {}
    finalist_count = {}
    semifinal_count = {}
    qf_count = {}

    print(f"\n  Simulating {n_sims:,} tournaments with {model_name} [{method}]...", flush=True)

    for i in range(n_sims):
        if (i + 1) % 2500 == 0:
            print(f"    {i + 1:,}/{n_sims:,}...", flush=True)

        champ, final, sf_losers, qf_winners = sim_tournament(
            model, get_lambdas, GROUPS_2026, rng, method, r, upset_sigma
        )

        champion_count[champ] = champion_count.get(champ, 0) + 1
        finalist_count[final] = finalist_count.get(final, 0) + 1
        finalist_count[champ] = finalist_count.get(champ, 0) + 1
        semifinal_count[champ] = semifinal_count.get(champ, 0) + 1
        semifinal_count[final] = semifinal_count.get(final, 0) + 1
        for t in sf_losers:
            semifinal_count[t] = semifinal_count.get(t, 0) + 1
        for t in qf_winners:
            qf_count[t] = qf_count.get(t, 0) + 1

    return champion_count, finalist_count, semifinal_count, qf_count, n_sims


def print_results(champion_count, finalist_count, semifinal_count, qf_count,
                  n_sims, model_name):
    """Print formatted results table."""
    print(f"\n{'=' * 72}")
    print(f"  {model_name}  ({n_sims:,} sims)")
    print(f"{'=' * 72}")
    print(f"  {'Team':<22} {'Champion':>10} {'Final':>10} {'Semi':>10} {'QF':>10}")
    print(f"  {'-' * 66}")

    sorted_teams = sorted(champion_count.keys(), key=lambda t: -champion_count[t])

    for team in sorted_teams[:25]:
        p_champ = champion_count.get(team, 0) / n_sims * 100
        p_final = finalist_count.get(team, 0) / n_sims * 100
        p_semi = semifinal_count.get(team, 0) / n_sims * 100
        p_qf = qf_count.get(team, 0) / n_sims * 100
        print(f"  {team:<22} {p_champ:>9.1f}% {p_final:>9.1f}% {p_semi:>9.1f}% {p_qf:>9.1f}%")


def print_comparison(poi_champ, nb_champ, n_sims):
    """Side-by-side comparison of Poisson vs Negative Binomial."""
    print(f"\n{'=' * 72}")
    print(f"  COMPARACION: Poisson vs Negative Binomial (Hybrid, {n_sims:,} sims)")
    print(f"{'=' * 72}")
    print(f"  {'Team':<22} {'Poisson':>10} {'Neg.Binom':>10} {'Diff':>10}")
    print(f"  {'-' * 56}")

    all_teams = set(list(poi_champ.keys()) + list(nb_champ.keys()))
    combined = {}
    for t in all_teams:
        p_poi = poi_champ.get(t, 0) / n_sims * 100
        p_nb = nb_champ.get(t, 0) / n_sims * 100
        combined[t] = (p_poi, p_nb)

    for team in sorted(combined, key=lambda t: -(combined[t][0] + combined[t][1]) / 2)[:25]:
        p_poi, p_nb = combined[team]
        diff = p_nb - p_poi
        arrow = ""
        if abs(diff) >= 0.5:
            arrow = " <--" if diff < 0 else " -->"
        print(f"  {team:<22} {p_poi:>9.1f}% {p_nb:>9.1f}% {diff:>+9.1f}%{arrow}")

    # Summary stats
    top6_poi = sorted(poi_champ.values(), reverse=True)[:6]
    top6_nb = sorted(nb_champ.values(), reverse=True)[:6]
    conc_poi = sum(top6_poi) / n_sims * 100
    conc_nb = sum(top6_nb) / n_sims * 100

    unique_champs_poi = sum(1 for v in poi_champ.values() if v > 0)
    unique_champs_nb = sum(1 for v in nb_champ.values() if v > 0)

    print(f"\n  {'Metrica':<35} {'Poisson':>10} {'Neg.Binom':>10}")
    print(f"  {'-' * 56}")
    print(f"  {'Top 6 concentracion':<35} {conc_poi:>9.1f}% {conc_nb:>9.1f}%")
    print(f"  {'Equipos distintos campeon':<35} {unique_champs_poi:>10} {unique_champs_nb:>10}")
    print(f"  {'Interpretacion':<35} {'predecible':>10} {'sorpresas':>10}")


def main():
    N_SIMS = 10000

    print("Loading data and training models...")
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")

    print("Training Hybrid model (Multi-Feature)...")
    hybrid_model = train_hybrid_model(df, cutoff)

    # === Poisson ===
    poi_champ, poi_final, poi_semi, poi_qf, n = run_montecarlo(
        hybrid_model, predict_lambdas_hybrid, "Hybrid + Poisson",
        n_sims=N_SIMS, seed=42, method="poisson"
    )
    print_results(poi_champ, poi_final, poi_semi, poi_qf, n, "Hybrid + Poisson")

    # === Negative Binomial ===
    nb_champ, nb_final, nb_semi, nb_qf, n = run_montecarlo(
        hybrid_model, predict_lambdas_hybrid, "Hybrid + Neg.Binomial (r=8)",
        n_sims=N_SIMS, seed=42, method="negbin", r=8.0
    )
    print_results(nb_champ, nb_final, nb_semi, nb_qf, n, "Hybrid + Neg.Binomial (r=8)")

    # === Comparison ===
    print_comparison(poi_champ, nb_champ, N_SIMS)


if __name__ == "__main__":
    main()
