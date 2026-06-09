"""Compare all models on WC backtests: Static (v0), Elo (v2.1), GAS (v2.2)."""
import numpy as np
from src.backtest import backtest, backtest_report
from src.metrics import rps


def _team_rps(bt, tournament_id, team):
    matches = bt[
        (bt["tournament_id"] == tournament_id)
        & ((bt["home_team"] == team) | (bt["away_team"] == team))
    ]
    if len(matches) == 0:
        return None, 0
    total = sum(
        rps(np.array([r["p_home"], r["p_draw"], r["p_away"]]), r["outcome"])
        for _, r in matches.iterrows()
    )
    return total / len(matches), len(matches)


def main():
    tournaments = ["WC2010", "WC2014", "WC2018", "WC2022"]
    models = [
        ("static", "Static (v0)"),
        ("elo", "Elo (v2.1)"),
        ("gas", "GAS (v2.2)"),
    ]

    print("=" * 78)
    print("MODEL COMPARISON: Static vs Elo vs GAS")
    print("=" * 78)

    results = {}
    bt_all = {}
    for model_type, label in models:
        print(f"\n>>> Running {label} backtest...")
        bt = backtest(tournaments, model_type=model_type)
        results[model_type] = backtest_report(bt)
        bt_all[model_type] = bt

    # Overall metrics
    print("\n" + "=" * 78)
    header = f"{'Metric':<15}"
    for _, label in models:
        header += f" {label:>16}"
    print(header)
    print("-" * 78)

    for metric in ["rps_mean", "log_loss", "brier", "ece"]:
        line = f"  {metric:<13}"
        vals = [results[m][metric] for m, _ in models]
        best = min(vals)
        for v in vals:
            marker = " *" if v == best else "  "
            line += f" {v:>14.4f}{marker}"
        print(line)

    # Per-tournament RPS
    print(f"\n--- PER TOURNAMENT RPS ---")
    header = f"{'Tournament':<12}"
    for _, label in models:
        header += f" {label:>16}"
    print(header)
    print("-" * 65)

    for tid in tournaments:
        line = f"  {tid:<10}"
        vals = [results[m][f"{tid}_rps"] for m, _ in models]
        best = min(vals)
        for v in vals:
            marker = " *" if v == best else "  "
            line += f" {v:>14.4f}{marker}"
        print(line)

    # Champion decline check
    print("\n--- CHAMPION DECLINE CHECK ---")
    for team, tid, context in [
        ("Spain", "WC2014", "defending champ, group stage exit"),
        ("Germany", "WC2018", "defending champ, group stage exit"),
    ]:
        print(f"\n  {team} at {tid} ({context}):")
        for model_type, label in models:
            avg, n = _team_rps(bt_all[model_type], tid, team)
            if avg is not None:
                print(f"    {label:<16}: avg RPS = {avg:.4f} ({n} matches)")


if __name__ == "__main__":
    main()
