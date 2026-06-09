"""Monte Carlo simulation of World Cup 2026 - champion probabilities."""

import math
import numpy as np
import pandas as pd
from src.data_loader import load_results
from src.strength_elo import train_elo_model
from src.strength_gas import train_gas_model
from src.wc2026 import GROUPS_2026
from run_wc2026 import (
    predict_lambdas_elo,
    predict_lambdas_gas,
    sample_scoreline,
    sample_knockout_scoreline,
)


def sim_group_stage_fast(model, get_lambdas, groups, rng):
    """Simulate group stage, return (standings dict, best 3rd place teams)."""
    standings = {}

    for gname, teams in groups.items():
        pts = [0] * 4
        gf = [0] * 4
        ga = [0] * 4

        for i in range(4):
            for j in range(i + 1, 4):
                lh, la = get_lambdas(model, teams[i], teams[j], neutral=True)
                gh, ga_ = sample_scoreline(rng, lh, la)
                gf[i] += gh; ga[i] += ga_
                gf[j] += ga_; ga[j] += gh
                if gh > ga_:
                    pts[i] += 3
                elif gh == ga_:
                    pts[i] += 1; pts[j] += 1
                else:
                    pts[j] += 3

        # Rank by pts, GD, GF, random
        indices = list(range(4))
        indices.sort(key=lambda k: (pts[k], gf[k] - ga[k], gf[k], rng.random()), reverse=True)
        standings[gname] = {
            "ranked": [teams[k] for k in indices],
            "pts": [pts[k] for k in indices],
            "gd": [gf[k] - ga[k] for k in indices],
            "gf": [gf[k] for k in indices],
        }

    # Best 3rd place
    thirds = []
    for g in sorted(standings):
        s = standings[g]
        thirds.append((g, s["ranked"][2], s["pts"][2], s["gd"][2], s["gf"][2]))
    thirds.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = [t[1] for t in thirds[:8]]

    return standings, best_thirds


def sim_knockout_match(model, get_lambdas, home, away, rng):
    """Simulate a knockout match, return winner."""
    lh, la = get_lambdas(model, home, away, neutral=True)
    gh, ga, _, _ = sample_knockout_scoreline(rng, lh, la)
    return home if gh > ga else away


def sim_tournament(model, get_lambdas, groups, rng):
    """Simulate full tournament, return (champion, finalist, semifinalists)."""
    standings, best_thirds = sim_group_stage_fast(model, get_lambdas, groups, rng)

    winners = {g: standings[g]["ranked"][0] for g in standings}
    runners = {g: standings[g]["ranked"][1] for g in standings}

    # Build R32 bracket
    tl = best_thirds  # list of 8 best 3rd-place teams
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

    # R32
    r32w = [sim_knockout_match(model, get_lambdas, h, a, rng) for h, a in r32]

    # R16
    r16 = [(r32w[i], r32w[i + 1]) for i in range(0, 16, 2)]
    r16w = [sim_knockout_match(model, get_lambdas, h, a, rng) for h, a in r16]

    # QF
    qf = [(r16w[i], r16w[i + 1]) for i in range(0, 8, 2)]
    qfw = [sim_knockout_match(model, get_lambdas, h, a, rng) for h, a in qf]

    # SF
    sf = [(qfw[0], qfw[1]), (qfw[2], qfw[3])]
    sfw = [sim_knockout_match(model, get_lambdas, h, a, rng) for h, a in sf]

    # Final
    champion = sim_knockout_match(model, get_lambdas, sfw[0], sfw[1], rng)
    finalist = sfw[1] if champion == sfw[0] else sfw[0]

    # Semifinalists (losers of SF)
    sf_losers = []
    for i, (h, a) in enumerate(sf):
        sf_losers.append(a if sfw[i] == h else h)

    return champion, finalist, sf_losers, qfw


def run_montecarlo(model, get_lambdas, model_name, n_sims=10000, seed=42):
    """Run Monte Carlo tournament simulation."""
    rng = np.random.default_rng(seed)

    champion_count = {}
    finalist_count = {}
    semifinal_count = {}
    qf_count = {}

    print(f"\n  Simulating {n_sims:,} tournaments with {model_name}...", flush=True)

    for i in range(n_sims):
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1:,}/{n_sims:,}...", flush=True)

        champ, final, sf_losers, qf_winners = sim_tournament(
            model, get_lambdas, GROUPS_2026, rng
        )

        champion_count[champ] = champion_count.get(champ, 0) + 1
        finalist_count[final] = finalist_count.get(final, 0) + 1
        finalist_count[champ] = finalist_count.get(champ, 0) + 1  # champion also reached final
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
    print(f"  WORLD CUP 2026 PROBABILITIES  --  {model_name}  ({n_sims:,} sims)")
    print(f"{'=' * 72}")
    print(f"  {'Team':<22} {'Champion':>10} {'Final':>10} {'Semi':>10} {'QF':>10}")
    print(f"  {'-' * 66}")

    # Sort by champion probability
    sorted_teams = sorted(champion_count.keys(), key=lambda t: -champion_count[t])

    for team in sorted_teams[:30]:
        p_champ = champion_count.get(team, 0) / n_sims * 100
        p_final = finalist_count.get(team, 0) / n_sims * 100
        p_semi = semifinal_count.get(team, 0) / n_sims * 100
        p_qf = qf_count.get(team, 0) / n_sims * 100
        print(f"  {team:<22} {p_champ:>9.1f}% {p_final:>9.1f}% {p_semi:>9.1f}% {p_qf:>9.1f}%")


def main():
    N_SIMS = 10000

    print("Loading data and training models...")
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")

    print("Training Elo model...")
    elo_model = train_elo_model(df, cutoff)

    print("Training GAS model...")
    gas_model = train_gas_model(df, cutoff)

    # Elo Monte Carlo
    elo_champ, elo_final, elo_semi, elo_qf, n = run_montecarlo(
        elo_model, predict_lambdas_elo, "Elo (v2.1)", n_sims=N_SIMS, seed=42
    )
    print_results(elo_champ, elo_final, elo_semi, elo_qf, n, "Elo (v2.1)")

    # GAS Monte Carlo
    gas_champ, gas_final, gas_semi, gas_qf, n = run_montecarlo(
        gas_model, predict_lambdas_gas, "GAS (v2.2)", n_sims=N_SIMS, seed=42
    )
    print_results(gas_champ, gas_final, gas_semi, gas_qf, n, "GAS (v2.2)")

    # Combined / Ensemble (average champion probabilities)
    print(f"\n{'=' * 72}")
    print(f"  ENSEMBLE (average of Elo + GAS)")
    print(f"{'=' * 72}")
    print(f"  {'Team':<22} {'Elo':>10} {'GAS':>10} {'Average':>10}")
    print(f"  {'-' * 56}")

    all_teams = set(list(elo_champ.keys()) + list(gas_champ.keys()))
    ensemble = {}
    for t in all_teams:
        p_elo = elo_champ.get(t, 0) / N_SIMS * 100
        p_gas = gas_champ.get(t, 0) / N_SIMS * 100
        ensemble[t] = (p_elo + p_gas) / 2

    for team in sorted(ensemble, key=lambda t: -ensemble[t])[:20]:
        p_elo = elo_champ.get(team, 0) / N_SIMS * 100
        p_gas = gas_champ.get(team, 0) / N_SIMS * 100
        avg = ensemble[team]
        print(f"  {team:<22} {p_elo:>9.1f}% {p_gas:>9.1f}% {avg:>9.1f}%")


if __name__ == "__main__":
    main()
