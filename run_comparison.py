"""Compare v0 Static model vs v2.1 Elo model on WC backtests."""
from src.backtest import backtest, backtest_report


def main():
    tournaments = ["WC2010", "WC2014", "WC2018", "WC2022"]

    print("=" * 65)
    print("MODEL COMPARISON: Static (v0) vs Elo (v2.1)")
    print("=" * 65)

    # Run static baseline
    print("\n>>> Running STATIC model backtest...")
    bt_static = backtest(tournaments, model_type="static")
    r_static = backtest_report(bt_static)

    # Run Elo model
    print("\n>>> Running ELO model backtest...")
    bt_elo = backtest(tournaments, model_type="elo")
    r_elo = backtest_report(bt_elo)

    # Comparison table
    print("\n" + "=" * 65)
    print(f"{'Metric':<20} {'Static (v0)':>15} {'Elo (v2.1)':>15} {'Delta':>10}")
    print("-" * 65)

    for metric in ["rps_mean", "log_loss", "brier", "ece"]:
        v0 = r_static[metric]
        v1 = r_elo[metric]
        delta = v1 - v0
        better = "<--" if delta < 0 else ""
        print(f"  {metric:<18} {v0:>15.4f} {v1:>15.4f} {delta:>+10.4f} {better}")

    print("\n--- PER TOURNAMENT RPS ---")
    print(f"{'Tournament':<12} {'Static':>10} {'Elo':>10} {'Delta':>10}")
    print("-" * 45)
    for tid in tournaments:
        v0 = r_static[f"{tid}_rps"]
        v1 = r_elo[f"{tid}_rps"]
        delta = v1 - v0
        better = "<--" if delta < 0 else ""
        print(f"  {tid:<10} {v0:>10.4f} {v1:>10.4f} {delta:>+10.4f} {better}")

    # Check champion decline test
    print("\n--- CHAMPION DECLINE CHECK ---")
    # Spain 2014 (defending WC2010): did Elo assign them lower prob?
    for bt, name in [(bt_static, "Static"), (bt_elo, "Elo")]:
        spain_matches = bt[
            (bt["tournament_id"] == "WC2014")
            & ((bt["home_team"] == "Spain") | (bt["away_team"] == "Spain"))
        ]
        if len(spain_matches) > 0:
            avg_rps = 0
            for _, row in spain_matches.iterrows():
                from src.metrics import rps
                import numpy as np
                probs = np.array([row["p_home"], row["p_draw"], row["p_away"]])
                avg_rps += rps(probs, row["outcome"])
            avg_rps /= len(spain_matches)
            print(f"  {name}: Spain WC2014 avg RPS = {avg_rps:.4f} ({len(spain_matches)} matches)")

    # Germany 2018
    for bt, name in [(bt_static, "Static"), (bt_elo, "Elo")]:
        ger_matches = bt[
            (bt["tournament_id"] == "WC2018")
            & ((bt["home_team"] == "Germany") | (bt["away_team"] == "Germany"))
        ]
        if len(ger_matches) > 0:
            avg_rps = 0
            for _, row in ger_matches.iterrows():
                from src.metrics import rps
                import numpy as np
                probs = np.array([row["p_home"], row["p_draw"], row["p_away"]])
                avg_rps += rps(probs, row["outcome"])
            avg_rps /= len(ger_matches)
            print(f"  {name}: Germany WC2018 avg RPS = {avg_rps:.4f} ({len(ger_matches)} matches)")


if __name__ == "__main__":
    main()
