import pandas as pd
import numpy as np
from src.data_loader import load_results, get_tournament_matches, WORLD_CUP_DATES
from src.strength_static import train_model as train_static
from src.metrics import match_outcome, rps, log_loss_1x2, brier_multi, calibration_ece


def backtest(
    tournament_ids: list[str],
    df: pd.DataFrame | None = None,
    model_type: str = "static",
) -> pd.DataFrame:
    """Walk-forward backtest over specified tournaments. Returns one row per match.

    model_type: "static" (v0 Poisson) or "elo" (Fase 2.1)
    """
    if df is None:
        df = load_results()

    if model_type == "elo":
        from src.strength_elo import train_elo_model
        train_fn = train_elo_model
    else:
        train_fn = train_static

    rows = []
    for tid in tournament_ids:
        info = WORLD_CUP_DATES[tid]
        cutoff = pd.Timestamp(info["cutoff"])
        print(f"  Training {model_type} model for {tid} (cutoff={cutoff.date()})...")
        model = train_fn(df, cutoff)
        matches = get_tournament_matches(df, info["name"], info["year"])
        print(f"  Found {len(matches)} matches for {tid}")

        for _, m in matches.iterrows():
            is_neutral = str(m.get("neutral", "TRUE")).upper() == "TRUE"
            probs = model.predict_match(m["home_team"], m["away_team"], neutral=is_neutral)
            outcome = match_outcome(int(m["home_score"]), int(m["away_score"]))

            rows.append({
                "tournament_id": tid,
                "match_date": m["date"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_score": int(m["home_score"]),
                "away_score": int(m["away_score"]),
                "outcome": outcome,
                "p_home": probs[0],
                "p_draw": probs[1],
                "p_away": probs[2],
                "train_cutoff": cutoff,
            })

    return pd.DataFrame(rows)


def backtest_report(bt: pd.DataFrame) -> dict:
    """Compute all metrics from backtest results."""
    probs = bt[["p_home", "p_draw", "p_away"]].values
    outcomes = bt["outcome"].values

    rps_scores = [rps(probs[i], outcomes[i]) for i in range(len(bt))]

    report = {
        "n_matches": len(bt),
        "rps_mean": float(np.mean(rps_scores)),
        "log_loss": log_loss_1x2(probs, outcomes),
        "brier": brier_multi(probs, outcomes),
        "ece": calibration_ece(probs, outcomes),
    }

    for tid in bt["tournament_id"].unique():
        mask = (bt["tournament_id"] == tid).values
        t_probs = probs[mask]
        t_outcomes = outcomes[mask]
        t_rps = [rps(t_probs[i], t_outcomes[i]) for i in range(mask.sum())]
        report[f"{tid}_rps"] = float(np.mean(t_rps))
        report[f"{tid}_log_loss"] = log_loss_1x2(t_probs, t_outcomes)
        report[f"{tid}_n"] = int(mask.sum())

    return report
