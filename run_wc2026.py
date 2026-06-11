"""Simulate the entire FIFA World Cup 2026 with match-by-match scorelines."""

import math
import numpy as np
from src.data_loader import load_results
from src.strength_elo import train_elo_model
from src.strength_gas import train_gas_model
from src.strength_hybrid import train_hybrid_model
from src.wc2026 import GROUPS_2026

# ---------------------------------------------------------------------------
# Poisson scoreline sampler
# ---------------------------------------------------------------------------

def predict_lambdas_elo(model, home, away, neutral=True):
    """Extract expected goals (lambdas) from Elo model."""
    r_home = model.ratings.get(home, 1500.0)
    r_away = model.ratings.get(away, 1500.0)
    diff = r_home - r_away + (model.home_adv if not neutral else 0)
    home_lambda = math.exp(0.25 + diff / model.scale)
    away_lambda = math.exp(0.25 - diff / model.scale)
    return max(0.2, min(home_lambda, 6.0)), max(0.2, min(away_lambda, 6.0))


def predict_lambdas_hybrid(model, home, away, neutral=True):
    """Extract expected goals (lambdas) from Hybrid model."""
    s_home = model.strength.get(home, 0.0)
    s_away = model.strength.get(away, 0.0)
    diff = s_home - s_away + (model.home_adv if not neutral else 0)
    # Head-to-head adjustment
    h2h_adj = model.h2h.get((home, away), 0.0) * model.h2h_weight
    diff += h2h_adj
    home_lambda = math.exp(0.25 + diff / model.scale)
    away_lambda = math.exp(0.25 - diff / model.scale)
    return max(0.2, min(home_lambda, 6.0)), max(0.2, min(away_lambda, 6.0))


def predict_lambdas_gas(model, home, away, neutral=True):
    """Extract expected goals (lambdas) from GAS model."""
    att_h = model.attack.get(home, 0.0)
    def_h = model.defense.get(home, 0.0)
    att_a = model.attack.get(away, 0.0)
    def_a = model.defense.get(away, 0.0)
    gamma = 0 if neutral else model.home_adv
    home_lambda = math.exp(model.intercept + att_h - def_a + gamma)
    away_lambda = math.exp(model.intercept + att_a - def_h)
    return max(0.2, min(home_lambda, 6.0)), max(0.2, min(away_lambda, 6.0))


def sample_scoreline(rng, home_lambda, away_lambda, method="poisson", r=8.0):
    """Sample a match scoreline.

    method="poisson": classic Poisson (variance = mean)
    method="negbin":  Negative Binomial (variance > mean, more upsets)
    r: dispersion parameter for negbin (lower = more variance)
    """
    if method == "negbin":
        h = int(rng.negative_binomial(r, r / (r + max(home_lambda, 0.15))))
        a = int(rng.negative_binomial(r, r / (r + max(away_lambda, 0.15))))
    else:
        h = int(rng.poisson(home_lambda))
        a = int(rng.poisson(away_lambda))
    return h, a


def sample_knockout_scoreline(rng, home_lambda, away_lambda, method="poisson", r=8.0):
    """Sample a knockout match scoreline. If draw after 90min, go to ET/pens."""
    h, a = sample_scoreline(rng, home_lambda, away_lambda, method, r)
    extra_time = False
    penalties = False

    if h == a:
        extra_time = True
        # Extra time: 1/3 of normal expected goals
        eh, ea = sample_scoreline(rng, home_lambda / 3, away_lambda / 3, method, r)
        h += eh
        a += ea

        if h == a:
            penalties = True
            # Penalty shootout: ~50-50 with slight edge to stronger team
            p_home = home_lambda / (home_lambda + away_lambda)
            p_home = 0.4 + 0.2 * p_home  # range 0.44-0.56
            if rng.random() < p_home:
                h += 1  # represent as +1 goal (pens)
            else:
                a += 1

    return h, a, extra_time, penalties


# ---------------------------------------------------------------------------
# Group stage simulation
# ---------------------------------------------------------------------------

def simulate_group_stage(model, get_lambdas, groups, rng):
    """Simulate all group matches. Returns match log and standings."""
    matches = []
    standings = {}

    for gname, teams in sorted(groups.items()):
        table = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0}
                 for t in teams}

        # Round-robin: each pair plays once
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home, away = teams[i], teams[j]
                lh, la = get_lambdas(model, home, away, neutral=True)
                probs = model.predict_match(home, away, neutral=True)
                gh, ga = sample_scoreline(rng, lh, la)

                matches.append({
                    "phase": "Group",
                    "group": gname,
                    "home": home,
                    "away": away,
                    "home_goals": gh,
                    "away_goals": ga,
                    "extra_time": False,
                    "penalties": False,
                    "p_home": probs[0],
                    "p_draw": probs[1],
                    "p_away": probs[2],
                })

                # Update table
                table[home]["gf"] += gh
                table[home]["ga"] += ga
                table[home]["gd"] += gh - ga
                table[away]["gf"] += ga
                table[away]["ga"] += gh
                table[away]["gd"] += ga - gh

                if gh > ga:
                    table[home]["pts"] += 3
                    table[home]["w"] += 1
                    table[away]["l"] += 1
                elif gh == ga:
                    table[home]["pts"] += 1
                    table[away]["pts"] += 1
                    table[home]["d"] += 1
                    table[away]["d"] += 1
                else:
                    table[away]["pts"] += 3
                    table[away]["w"] += 1
                    table[home]["l"] += 1

        # Sort: points, then GD, then GF, then random tiebreak
        ranked = sorted(
            teams,
            key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"], rng.random()),
            reverse=True,
        )
        standings[gname] = [(t, table[t]) for t in ranked]

    return matches, standings


# ---------------------------------------------------------------------------
# Best 3rd-place teams selection
# ---------------------------------------------------------------------------

def select_best_thirds(standings):
    """Select the 8 best 3rd-placed teams from 12 groups."""
    thirds = []
    for gname, ranked in sorted(standings.items()):
        team, stats = ranked[2]  # 3rd place
        thirds.append((gname, team, stats))

    # Sort by points, GD, GF
    thirds.sort(key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"]), reverse=True)
    return thirds[:8]  # top 8


# ---------------------------------------------------------------------------
# Knockout bracket
# ---------------------------------------------------------------------------

def build_r32_bracket(standings, best_thirds):
    """Build the Round of 32 matchups."""
    # Get teams by position
    winners = {g: standings[g][0][0] for g in standings}    # 1st place
    runners = {g: standings[g][1][0] for g in standings}    # 2nd place
    third_teams = {g: team for g, team, _ in best_thirds}   # 3rd place (group -> team)

    # Assign 3rd-place teams to bracket slots based on which groups they came from
    third_list = list(third_teams.values())

    # R32 bracket per official FIFA 2026 knockout structure (M73-M88).
    # Consecutive pairs feed into same R16 match.
    tl = third_list
    bracket = [
        # --- Left half -> QF1, QF2 -> SF1 ---
        # R16-1 (M89): W73 vs W77
        (runners["A"], runners["B"]),                          # M73: 2A vs 2B
        (winners["I"], tl[0] if len(tl) > 0 else "TBD"),      # M77: 1I vs 3rd
        # R16-2 (M90): W74 vs W75
        (winners["E"], tl[1] if len(tl) > 1 else "TBD"),      # M74: 1E vs 3rd
        (winners["C"], runners["F"]),                          # M75: 1C vs 2F
        # R16-3 (M91): W76 vs W78
        (winners["F"], runners["C"]),                          # M76: 1F vs 2C
        (runners["E"], runners["I"]),                          # M78: 2E vs 2I
        # R16-4 (M92): W79 vs W80
        (winners["A"], tl[2] if len(tl) > 2 else "TBD"),      # M79: 1A vs 3rd
        (winners["D"], tl[3] if len(tl) > 3 else "TBD"),      # M80: 1D vs 3rd
        # --- Right half -> QF3, QF4 -> SF2 ---
        # R16-5 (M93): W81 vs W82
        (winners["G"], tl[4] if len(tl) > 4 else "TBD"),      # M81: 1G vs 3rd
        (winners["L"], tl[5] if len(tl) > 5 else "TBD"),      # M82: 1L vs 3rd
        # R16-6 (M94): W83 vs W84
        (winners["B"], tl[6] if len(tl) > 6 else "TBD"),      # M83: 1B vs 3rd
        (winners["K"], tl[7] if len(tl) > 7 else "TBD"),      # M84: 1K vs 3rd
        # R16-7 (M95): W85 vs W86
        (winners["H"], runners["J"]),                          # M85: 1H vs 2J
        (winners["J"], runners["H"]),                          # M86: 1J vs 2H
        # R16-8 (M96): W87 vs W88
        (runners["D"], runners["G"]),                          # M87: 2D vs 2G
        (runners["K"], runners["L"]),                          # M88: 2K vs 2L
    ]

    return bracket


def simulate_knockout_round(model, get_lambdas, matchups, round_name, rng):
    """Simulate one knockout round. Returns match log and winners."""
    matches = []
    winners = []

    for home, away in matchups:
        lh, la = get_lambdas(model, home, away, neutral=True)
        probs = model.predict_match(home, away, neutral=True)
        gh, ga, et, pens = sample_knockout_scoreline(rng, lh, la)

        winner = home if gh > ga else away

        matches.append({
            "phase": round_name,
            "group": "",
            "home": home,
            "away": away,
            "home_goals": gh,
            "away_goals": ga,
            "extra_time": et,
            "penalties": pens,
            "p_home": probs[0],
            "p_draw": probs[1],
            "p_away": probs[2],
        })
        winners.append(winner)

    return matches, winners


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_group_standings(standings):
    print("\n" + "=" * 75)
    print("GROUP STANDINGS")
    print("=" * 75)

    for gname in sorted(standings.keys()):
        print(f"\n  Group {gname}")
        print(f"  {'Team':<25} {'Pts':>4} {'W':>3} {'D':>3} {'L':>3} {'GF':>4} {'GA':>4} {'GD':>4}")
        print(f"  {'-'*52}")
        for i, (team, s) in enumerate(standings[gname]):
            marker = " Q" if i < 2 else (" ?" if i == 2 else "  ")
            print(f"  {team:<25} {s['pts']:>4} {s['w']:>3} {s['d']:>3} {s['l']:>3} "
                  f"{s['gf']:>4} {s['ga']:>4} {s['gd']:>+4}{marker}")


def print_matches(matches, title):
    print(f"\n{'=' * 75}")
    print(title)
    print("=" * 75)

    current_group = None
    for m in matches:
        if m["phase"] == "Group" and m["group"] != current_group:
            current_group = m["group"]
            print(f"\n  --- Group {current_group} ---")

        home = m["home"]
        away = m["away"]
        gh = m["home_goals"]
        ga = m["away_goals"]

        # Score display
        score = f"{gh} - {ga}"
        if m["penalties"]:
            score += " (pens)"
        elif m["extra_time"]:
            score += " (a.e.t.)"

        # Upset indicator
        p_home = m["p_home"]
        p_away = m["p_away"]
        if gh > ga and p_away > 0.45:
            upset = " <!>"
        elif ga > gh and p_home > 0.45:
            upset = " <!>"
        else:
            upset = ""

        print(f"  {home:<22} {score:^11} {away:>22}  "
              f"({p_home:.0%}/{m['p_draw']:.0%}/{p_away:.0%}){upset}")


def print_bracket(rounds_data):
    """Print knockout bracket in a readable format."""
    for round_name, matches, winners in rounds_data:
        print(f"\n{'=' * 75}")
        print(f"  {round_name}")
        print(f"{'=' * 75}")
        for m in matches:
            home = m["home"]
            away = m["away"]
            gh = m["home_goals"]
            ga = m["away_goals"]

            score = f"{gh} - {ga}"
            if m["penalties"]:
                score += " (pens)"
            elif m["extra_time"]:
                score += " (a.e.t.)"

            winner = home if gh > ga else away
            print(f"  {home:<22} {score:^15} {away:>22}  -> {winner}")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def check_teams_exist(model, groups):
    """Check which teams are missing from the model and warn."""
    missing = []
    for gname, teams in groups.items():
        for t in teams:
            if hasattr(model, 'ratings'):
                if t not in model.ratings:
                    missing.append(t)
            elif hasattr(model, 'attack'):
                if t not in model.attack:
                    missing.append(t)
    if missing:
        print(f"\n  WARNING: {len(missing)} teams not found in model history:")
        for t in sorted(set(missing)):
            print(f"    - {t}")
        print("  These teams will get default/average strength.\n")


def run_simulation(model, get_lambdas, model_name, seed=2026):
    rng = np.random.default_rng(seed)

    print(f"\n{'#' * 75}")
    print(f"#  FIFA WORLD CUP 2026 SIMULATION  --  Model: {model_name}")
    print(f"#  Seed: {seed}")
    print(f"{'#' * 75}")

    check_teams_exist(model, GROUPS_2026)

    # Group stage
    group_matches, standings = simulate_group_stage(model, get_lambdas, GROUPS_2026, rng)
    print_matches(group_matches, "GROUP STAGE RESULTS")
    print_group_standings(standings)

    # Best 3rd-place teams
    best_thirds = select_best_thirds(standings)
    print(f"\n{'=' * 75}")
    print("BEST 3RD-PLACED TEAMS (qualified)")
    print("=" * 75)
    for g, team, s in best_thirds:
        print(f"  Group {g}: {team:<25} {s['pts']}pts  GD={s['gd']:+d}")

    # Build knockout bracket
    r32_matchups = build_r32_bracket(standings, best_thirds)

    # R32
    r32_matches, r32_winners = simulate_knockout_round(
        model, get_lambdas, r32_matchups, "Round of 32", rng)

    # R16
    r16_matchups = [(r32_winners[i], r32_winners[i + 1]) for i in range(0, 16, 2)]
    r16_matches, r16_winners = simulate_knockout_round(
        model, get_lambdas, r16_matchups, "Round of 16", rng)

    # QF
    qf_matchups = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, 8, 2)]
    qf_matches, qf_winners = simulate_knockout_round(
        model, get_lambdas, qf_matchups, "Quarter-Finals", rng)

    # SF
    sf_matchups = [(qf_winners[i], qf_winners[i + 1]) for i in range(0, 4, 2)]
    sf_matches, sf_winners = simulate_knockout_round(
        model, get_lambdas, sf_matchups, "Semi-Finals", rng)

    # 3rd place
    losers = []
    for m in sf_matches:
        loser = m["away"] if m["home_goals"] > m["away_goals"] else m["home"]
        losers.append(loser)
    third_matchup = [(losers[0], losers[1])]
    third_matches, third_winner = simulate_knockout_round(
        model, get_lambdas, third_matchup, "3rd Place Match", rng)

    # Final
    final_matchup = [(sf_winners[0], sf_winners[1])]
    final_matches, champion = simulate_knockout_round(
        model, get_lambdas, final_matchup, "FINAL", rng)

    # Print knockout bracket
    rounds = [
        ("ROUND OF 32", r32_matches, r32_winners),
        ("ROUND OF 16", r16_matches, r16_winners),
        ("QUARTER-FINALS", qf_matches, qf_winners),
        ("SEMI-FINALS", sf_matches, sf_winners),
        ("3RD PLACE MATCH", third_matches, third_winner),
        ("FINAL", final_matches, champion),
    ]
    print_bracket(rounds)

    # Final podium
    m = final_matches[0]
    final_winner = m["home"] if m["home_goals"] > m["away_goals"] else m["away"]
    final_loser = m["away"] if m["home_goals"] > m["away_goals"] else m["home"]

    print(f"\n{'*' * 75}")
    print(f"*")
    print(f"*  CHAMPION:    {final_winner}")
    print(f"*  RUNNER-UP:   {final_loser}")
    print(f"*  3RD PLACE:   {third_winner[0]}")
    print(f"*")
    print(f"{'*' * 75}")

    # Match statistics
    all_matches = group_matches + r32_matches + r16_matches + qf_matches + sf_matches + third_matches + final_matches
    total_goals = sum(m["home_goals"] + m["away_goals"] for m in all_matches)
    print(f"\n  Total matches: {len(all_matches)}")
    print(f"  Total goals: {total_goals}")
    print(f"  Goals per match: {total_goals / len(all_matches):.2f}")

    return final_winner


def main():
    print("Loading data and training models...")
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")  # WC2026 start date

    print("Training Hybrid model (Multi-Feature, 14 signals)...")
    hybrid_model = train_hybrid_model(df, cutoff)

    # Print top 25 blended strength
    print("\n" + "=" * 55)
    print("TOP 25 BLENDED STRENGTH (Multi-Feature)")
    print("=" * 55)
    ranked = sorted(hybrid_model.strength.items(), key=lambda x: -x[1])[:25]
    for i, (team, s) in enumerate(ranked, 1):
        print(f"  {i:>2}. {team:<25} {s:>+.3f}")

    # Simulate multiple seeds for variety
    seeds = [2026, 7, 42, 1986, 2010]
    champions = {}
    for seed in seeds:
        champ = run_simulation(hybrid_model, predict_lambdas_hybrid,
                               "Hybrid (Multi-Feature)", seed=seed)
        champions[seed] = champ

    print(f"\n\n{'=' * 75}")
    print("RESUMEN - 5 SIMULACIONES")
    print("=" * 75)
    for seed, champ in champions.items():
        print(f"  Seed {seed:<6} -> Campeon: {champ}")
    from collections import Counter
    counts = Counter(champions.values())
    print(f"\n  Campeon mas frecuente: {counts.most_common(1)[0][0]} ({counts.most_common(1)[0][1]}/5)")


if __name__ == "__main__":
    import pandas as pd
    main()
