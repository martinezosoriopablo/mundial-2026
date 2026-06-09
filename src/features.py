"""Feature engineering for World Cup prediction.

Computes team-level features from multiple data sources:
- Momentum: win % in last N matches
- Defensive strength: goals conceded per match (last 10)
- League composite: weighted score based on where players play
- Defending champion: binary flag for last WC winner
- Squad age: distance from optimal age (penalty for too young/old)
- Coach tenure: years in charge (sweet spot 2-8 years)
- Host advantage: binary flag for host nations
- Population: log-scaled (proxy for talent pool depth)
- Market odds: bookmaker implied probabilities
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
LEAGUE_PLAYERS_PATH = DATA_DIR / "league_players.csv"
BETTING_ODDS_PATH = DATA_DIR / "betting_odds.csv"
SQUAD_AGE_PATH = DATA_DIR / "squad_age.csv"
COACH_TENURE_PATH = DATA_DIR / "coach_tenure.csv"
POPULATION_PATH = DATA_DIR / "population.csv"
DIVERSITY_PATH = DATA_DIR / "squad_diversity.csv"

LEAGUE_WEIGHTS = {
    "england": 1.00, "spain": 0.95, "italy": 0.85, "germany": 0.80,
    "france": 0.75, "portugal": 0.60, "netherlands": 0.50,
    "brazil": 0.45, "argentina": 0.40, "other_europe": 0.35, "domestic": 0.10,
}

# WC2026 host nations
HOST_NATIONS = {"United States", "Mexico", "Canada"}

OPTIMAL_AGE = 26.9  # average age of last 10 WC champions


def _zscore(d: dict[str, float]) -> dict[str, float]:
    """Z-score normalize a dict of values."""
    vals = np.array(list(d.values()))
    mean, std = vals.mean(), vals.std()
    if std < 1e-8:
        std = 1.0
    return {t: (v - mean) / std for t, v in d.items()}


def compute_momentum(df: pd.DataFrame, cutoff: pd.Timestamp,
                     n_matches: int = 10,
                     elo_ratings: dict[str, float] | None = None) -> dict[str, float]:
    """Compute Elo-weighted momentum in last N matches before cutoff.

    Each result is weighted by opponent Elo:
      - Beat a top team (Elo 2100+) → weight ~1.4
      - Beat an average team (Elo 1500) → weight ~1.0
      - Beat a weak team (Elo 1200) → weight ~0.8
    This prevents friendly wins vs Iceland/Ghana from inflating momentum.
    """
    df_pre = df[df["date"] < cutoff].copy()
    all_teams = set(df_pre["home_team"].unique()) | set(df_pre["away_team"].unique())
    momentum = {}

    # If no Elo provided, use uniform weights (backward compatible)
    elo_center = 1500.0
    if elo_ratings:
        elo_vals = list(elo_ratings.values())
        elo_center = sum(elo_vals) / len(elo_vals)

    for team in all_teams:
        mask = (df_pre["home_team"] == team) | (df_pre["away_team"] == team)
        team_matches = df_pre[mask].tail(n_matches)

        if len(team_matches) == 0:
            momentum[team] = 0.5
            continue

        weighted_points = 0.0
        total_weight = 0.0
        for _, r in team_matches.iterrows():
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            is_home = r["home_team"] == team
            opponent = r["away_team"] if is_home else r["home_team"]

            # Opponent quality weight: Elo / center_Elo (clamped 0.6-1.6)
            if elo_ratings and opponent in elo_ratings:
                opp_elo = elo_ratings[opponent]
                w = max(0.6, min(1.6, opp_elo / elo_center))
            else:
                w = 1.0

            # Tournament weight: competitive > friendly
            tourn = str(r.get("tournament", "")).lower()
            is_competitive = any(kw in tourn for kw in [
                "world cup", "euro", "copa am", "nations league",
                "qualif", "africa cup", "asian cup", "gold cup"
            ])
            if not is_competitive:
                w *= 0.6  # friendlies count 60% as much

            if is_home:
                if hs > as_: weighted_points += 1.0 * w
                elif hs == as_: weighted_points += 0.5 * w
            else:
                if as_ > hs: weighted_points += 1.0 * w
                elif hs == as_: weighted_points += 0.5 * w

            total_weight += w

        momentum[team] = weighted_points / total_weight if total_weight > 0 else 0.5

    return momentum


def compute_defensive_strength(df: pd.DataFrame, cutoff: pd.Timestamp,
                                n_matches: int = 10) -> dict[str, float]:
    """Compute goals conceded per match in last N matches (lower = better).

    Returns NEGATIVE z-score (so higher = better defense for blending).
    """
    df_pre = df[df["date"] < cutoff].copy()
    all_teams = set(df_pre["home_team"].unique()) | set(df_pre["away_team"].unique())
    defense = {}

    for team in all_teams:
        mask = (df_pre["home_team"] == team) | (df_pre["away_team"] == team)
        team_matches = df_pre[mask].tail(n_matches)

        if len(team_matches) == 0:
            defense[team] = 1.5  # average
            continue

        conceded = 0
        for _, r in team_matches.iterrows():
            if r["home_team"] == team:
                conceded += int(r["away_score"])
            else:
                conceded += int(r["home_score"])

        defense[team] = conceded / len(team_matches)

    # Invert: fewer goals conceded = higher strength
    return {t: -v for t, v in defense.items()}


def compute_league_composite(path: Path = LEAGUE_PLAYERS_PATH) -> dict[str, float]:
    """Weighted league quality composite with domestic discount (z-scored).

    Key insight: an English player in the Premier League is the baseline —
    they play there by default (language, culture, proximity). A Moroccan
    in the PL had to be BETTER than English players to earn that spot
    (selection bias). So we discount players who play in their OWN
    country's top league — it's less informative about their quality.

    domestic_ratio = players in own country's top league / total squad
    discount = 1 - domestic_ratio * 0.40  (40% haircut for domestic players)
    """
    # Map national team -> their own league column
    TEAM_TO_OWN_LEAGUE = {
        "France": "france", "England": "england", "Spain": "spain",
        "Germany": "germany", "Italy": "italy", "Portugal": "portugal",
        "Netherlands": "netherlands", "Brazil": "brazil", "Argentina": "argentina",
    }
    DOMESTIC_DISCOUNT = 0.40  # 40% haircut

    df = pd.read_csv(path)
    composites = {}
    for _, row in df.iterrows():
        team = row["team"]
        total = row["total_squad"]
        weighted_sum = sum(row.get(lg, 0) * w for lg, w in LEAGUE_WEIGHTS.items()
                          if lg in row.index)
        raw_composite = weighted_sum / total

        # Domestic discount: if team has a top league, discount players
        # who play in their own league (less signal about quality)
        own_league = TEAM_TO_OWN_LEAGUE.get(team)
        if own_league and own_league in row.index:
            domestic_in_own = row[own_league]
            domestic_ratio = domestic_in_own / total
            discount = 1 - domestic_ratio * DOMESTIC_DISCOUNT
            raw_composite *= discount

        composites[team] = raw_composite
    return _zscore(composites)


def get_defending_champion(cutoff: pd.Timestamp) -> str | None:
    """Return the defending World Cup champion."""
    wc_winners = {
        2002: "Brazil", 2006: "Italy", 2010: "Spain",
        2014: "Germany", 2018: "France", 2022: "Argentina",
    }
    for year in sorted(wc_winners.keys(), reverse=True):
        if pd.Timestamp(f"{year}-12-31") < cutoff:
            return wc_winners[year]
    return None


def compute_defending_champion_feature(cutoff: pd.Timestamp,
                                       all_teams: set[str]) -> dict[str, float]:
    """1.0 for defending champion, 0.0 for others."""
    champ = get_defending_champion(cutoff)
    return {t: (1.0 if t == champ else 0.0) for t in all_teams}


def compute_age_fitness(path: Path = SQUAD_AGE_PATH) -> dict[str, float]:
    """Compute age fitness: penalty for distance from optimal age.

    Uses a Gaussian-like curve centered at OPTIMAL_AGE (26.9).
    Teams at the optimal age get the highest score.
    Returns z-scored values.
    """
    df = pd.read_csv(path)
    sigma = 2.0  # how quickly fitness drops off from optimal
    df["fitness"] = np.exp(-0.5 * ((df["avg_age"] - OPTIMAL_AGE) / sigma) ** 2)
    return _zscore(dict(zip(df["team"], df["fitness"])))


def compute_coach_tenure(path: Path = COACH_TENURE_PATH) -> dict[str, float]:
    """Compute coach tenure fitness score.

    Historical sweet spot for WC winners: 2-8 years.
    <1 year: penalty (not enough time to build)
    2-8 years: optimal
    >10 years: slight penalty (staleness)
    """
    df = pd.read_csv(path)

    def tenure_score(years):
        if years < 1:
            return 0.3
        elif years <= 2:
            return 0.7
        elif years <= 8:
            return 1.0
        elif years <= 12:
            return 0.8
        else:
            return 0.6  # very long tenure, possible staleness

    df["score"] = df["tenure_years"].apply(tenure_score)
    return _zscore(dict(zip(df["team"], df["score"])))


def compute_host_advantage(all_teams: set[str]) -> dict[str, float]:
    """Host advantage flag (z-scored for blending)."""
    raw = {t: (1.0 if t in HOST_NATIONS else 0.0) for t in all_teams}
    return _zscore(raw)


def compute_population_factor(path: Path = POPULATION_PATH) -> dict[str, float]:
    """Log-scaled population as talent pool proxy (z-scored).

    Larger populations have deeper talent pools, but with diminishing returns.
    """
    df = pd.read_csv(path)
    df["log_pop"] = np.log(df["population_m"])
    return _zscore(dict(zip(df["team"], df["log_pop"])))


def compute_market_strength(path: Path = BETTING_ODDS_PATH) -> dict[str, float]:
    """Convert bookmaker odds to implied strength (z-score normalized)."""
    df = pd.read_csv(path)
    df["implied_prob"] = 100.0 / (df["american_odds"] + 100.0)
    total_prob = df["implied_prob"].sum()
    df["fair_prob"] = df["implied_prob"] / total_prob
    df["logit"] = np.log(df["fair_prob"] / (1 - df["fair_prob"]))
    return _zscore(dict(zip(df["team"], df["logit"])))


def compute_diversity_index(path: Path = DIVERSITY_PATH) -> dict[str, float]:
    """Compute squad diversity as proxy for recruitment breadth.

    NOT a racial variable — measures the effective talent pool breadth.
    Countries that recruit from diverse diasporas (France, England, Belgium,
    Netherlands) have a wider talent pool than their population alone suggests.

    The key insight: France 2018 (7/11 African descent, champion) vs
    Spain 2010 (0/11, champion) — diversity per se doesn't predict winning.
    But diversity IN COMBINATION with a top-tier league system does.

    We model this as: diversity * league_quality interaction.
    High diversity + top league = huge advantage (France, England)
    High diversity + weak league = no advantage (Haiti, Curaçao)
    """
    df = pd.read_csv(path)

    # Load league composite for interaction
    league = compute_league_composite()

    diversity_scores = {}
    for _, row in df.iterrows():
        team = row["team"]
        # Dual heritage ratio: what fraction of squad has multinational heritage
        dual_ratio = row["dual_heritage_players"] / row["total_squad"]

        # Pure diversity doesn't help — it needs to interact with league quality
        # France: high dual_ratio (0.69) * high league (+2.26) = strong
        # Haiti: high dual_ratio (1.0) * low league (-1.5) = weak
        lg_score = league.get(team, 0.0)

        # Interaction: diversity only boosts if league composite is positive
        if lg_score > 0:
            diversity_scores[team] = dual_ratio * lg_score
        else:
            diversity_scores[team] = 0.0  # no boost for weak-league diverse teams

    return _zscore(diversity_scores)


def compute_composition_threshold(path: Path = DIVERSITY_PATH) -> dict[str, float]:
    """Historical threshold: no champion has ever had 6+ Black starters in XI.

    Binary penalty for teams where 8+ of 11 starters would be Black.
    This primarily affects African national teams (11/11) and Caribbean teams,
    NOT European teams with diverse squads (France typically starts ~5).

    The threshold at 8/11 (73%+) reflects:
    - France 2018 won with 5/11 (~45%) - no penalty
    - No African team (11/11, 100%) has passed QF (except Morocco, Arab/Berber)
    - The pattern is about teams with near-100% composition, not mixed teams
    """
    df = pd.read_csv(path)
    scores = {}
    for _, row in df.iterrows():
        team = row["team"]
        pct = row["players_african_descent"] / row["total_squad"]

        if pct >= 0.73:  # 73%+ of squad -> likely 8+/11 Black starters
            scores[team] = -1.0  # penalty
        else:
            scores[team] = 0.0

    return _zscore(scores)


def compute_frontrunner_curse(path: Path = BETTING_ODDS_PATH) -> dict[str, float]:
    """The frontrunner curse: #1 pre-tournament favorite wins only 14% of the time.

    Historical pattern (1998-2022):
      #1 favorite: 1/7 wins (14%) - heavy penalty
      #2 favorite: 0/7 wins (0%)  - moderate penalty
      #3 favorite: 5/7 wins (71%) - SWEET SPOT boost
      #4-#6:       moderate boost
      #7+:         neutral

    This captures pressure, over-preparation by opponents, complacency.
    """
    df = pd.read_csv(path)
    df["implied_prob"] = 100.0 / (df["american_odds"] + 100.0)
    df = df.sort_values("implied_prob", ascending=False).reset_index(drop=True)

    # Rank-based adjustment
    rank_scores = {
        0: -0.8,   # #1 favorite: strong penalty (14% historical win rate)
        1: -0.3,   # #2: moderate penalty
        2:  0.8,   # #3: SWEET SPOT (71% historical win rate!)
        3:  0.5,   # #4: good position
        4:  0.3,   # #5: decent
        5:  0.2,   # #6: slightly positive
    }

    scores = {}
    for i, row in df.iterrows():
        scores[row["team"]] = rank_scores.get(i, 0.0)

    return _zscore(scores)
