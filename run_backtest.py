"""Run the full Phase 1 backtest and print results."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.backtest import backtest, backtest_report
from src.metrics import rps


def plot_calibration(bt, output_path="calibration.png"):
    """Plot reliability diagram."""
    probs = bt[["p_home", "p_draw", "p_away"]].values
    outcomes = bt["outcome"].values
    n_bins = 10

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["Home Win", "Draw", "Away Win"]

    for cls in range(3):
        p = probs[:, cls]
        y = (outcomes == cls).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_mids = []
        bin_freqs = []
        bin_counts = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() > 0:
                bin_mids.append(p[mask].mean())
                bin_freqs.append(y[mask].mean())
                bin_counts.append(mask.sum())

        ax = axes[cls]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.scatter(bin_mids, bin_freqs, s=[c * 3 for c in bin_counts], alpha=0.7)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{labels[cls]}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.suptitle("Calibration (Reliability Diagram) - v0 Static Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nCalibration plot saved to {output_path}")


def main():
    print("=" * 60)
    print("PHASE 1 BACKTEST - v0 Static Poisson Model")
    print("=" * 60)

    tournaments = ["WC2010", "WC2014", "WC2018", "WC2022"]

    print(f"\nRunning walk-forward backtest on: {tournaments}")
    bt = backtest(tournaments)

    print(f"\nTotal matches predicted: {len(bt)}")
    report = backtest_report(bt)

    print("\n--- OVERALL METRICS ---")
    print(f"  RPS (mean):  {report['rps_mean']:.4f}")
    print(f"  Log-loss:    {report['log_loss']:.4f}")
    print(f"  Brier:       {report['brier']:.4f}")
    print(f"  ECE:         {report['ece']:.4f}")

    print("\n--- PER TOURNAMENT ---")
    for tid in tournaments:
        print(
            f"  {tid}: RPS={report[f'{tid}_rps']:.4f}  "
            f"LogLoss={report[f'{tid}_log_loss']:.4f}  "
            f"N={report[f'{tid}_n']}"
        )

    plot_calibration(bt)

    bt.to_csv("backtest_results.csv", index=False)
    print(f"Detailed results saved to backtest_results.csv")


if __name__ == "__main__":
    main()
