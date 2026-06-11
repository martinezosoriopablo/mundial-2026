"""Grid search upset_sigma to calibrate champion probabilities to market odds.

With scale floor removed (using MLE-optimal scale), we need a higher upset_sigma
to spread out champion probabilities and match market expectations.
"""
import numpy as np
import pandas as pd
from src.data_loader import load_results
from src.strength_hybrid import train_hybrid_model
from run_wc2026 import predict_lambdas_hybrid
from run_montecarlo import run_montecarlo

MARKET = {
    "Spain": 15.8, "France": 14.9, "England": 10.8,
    "Brazil": 9.1, "Portugal": 9.1, "Argentina": 8.7,
    "Germany": 5.8, "Netherlands": 4.1, "Norway": 2.4,
    "Belgium": 2.1, "Colombia": 2.1, "Japan": 1.9,
    "Croatia": 1.7, "Switzerland": 1.7, "Morocco": 1.7,
    "Mexico": 1.5, "United States": 1.4, "Uruguay": 1.3,
    "Turkey": 1.1,
}

def main():
    print("Loading data and training model...")
    df = load_results()
    cutoff = pd.Timestamp("2026-06-11")
    model = train_hybrid_model(df, cutoff)

    print(f"\nModel scale (MLE): {model.scale:.2f}")
    print(f"Top strengths: Spain={model.strength.get('Spain',0):.3f}, "
          f"France={model.strength.get('France',0):.3f}, "
          f"Argentina={model.strength.get('Argentina',0):.3f}")

    # Quick example: Spain vs Cape Verde lambdas
    import math
    for h, a in [("Spain", "Cape Verde"), ("Germany", "Curaçao"), ("England", "Ghana")]:
        lh, la = predict_lambdas_hybrid(model, h, a, neutral=True)
        print(f"  {h} vs {a}: {lh:.1f} - {la:.1f}")

    # Grid search upset_sigma
    sigmas = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    N_SIMS = 10000  # more sims for fine-tuning

    print(f"\n{'='*80}")
    print(f"  GRID SEARCH: upset_sigma (N={N_SIMS:,} sims each)")
    print(f"{'='*80}")

    best_sigma = 0.15
    best_rmse = 999

    for sigma in sigmas:
        champ, _, _, _, n = run_montecarlo(
            model, predict_lambdas_hybrid, f"sigma={sigma:.2f}",
            n_sims=N_SIMS, seed=42, method="poisson", upset_sigma=sigma
        )

        # Calculate RMSE vs market for top teams
        errors = []
        for team, mkt_pct in MARKET.items():
            model_pct = champ.get(team, 0) / n * 100
            errors.append((model_pct - mkt_pct) ** 2)
        rmse = np.sqrt(np.mean(errors))

        # Top 6 concentration
        top6 = sorted(champ.values(), reverse=True)[:6]
        conc = sum(top6) / n * 100

        # Top 3 teams
        top3 = sorted(champ.keys(), key=lambda t: -champ[t])[:3]
        top3_str = ", ".join(f"{t}={champ[t]/n*100:.1f}%" for t in top3)

        print(f"\n  sigma={sigma:.2f} | RMSE={rmse:.2f}pp | Top6={conc:.1f}% | {top3_str}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_sigma = sigma

    print(f"\n{'='*80}")
    print(f"  BEST: upset_sigma={best_sigma:.2f} (RMSE={best_rmse:.2f}pp)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
