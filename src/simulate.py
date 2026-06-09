import numpy as np
from src.strength_static import StaticModel


def simulate_group(model: StaticModel, teams: list[str], rng=None) -> list[str]:
    """Simulate round-robin group, return top 2 teams by points (random tiebreak)."""
    if rng is None:
        rng = np.random.default_rng()

    points = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            probs = model.predict_match(teams[i], teams[j], neutral=True)
            outcome = rng.choice([0, 1, 2], p=probs)
            if outcome == 0:
                points[teams[i]] += 3
            elif outcome == 1:
                points[teams[i]] += 1
                points[teams[j]] += 1
            else:
                points[teams[j]] += 3

    ranked = sorted(teams, key=lambda t: (points[t], rng.random()), reverse=True)
    return ranked[:2]


def simulate_knockout(model: StaticModel, teams: list[str], rng=None) -> str:
    """Single-elimination bracket. teams length must be power of 2."""
    if rng is None:
        rng = np.random.default_rng()

    current = list(teams)
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            probs = model.predict_match(current[i], current[i + 1], neutral=True)
            # In knockout, draw goes to extra time/pens: redistribute draw prob
            p_h = probs[0] + probs[1] * probs[0] / (probs[0] + probs[2])
            winner = current[i] if rng.random() < p_h else current[i + 1]
            next_round.append(winner)
        current = next_round
    return current[0]


def simulate_tournament(
    model: StaticModel,
    groups: dict[str, list[str]],
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Monte Carlo tournament simulation. Returns {team: P(champion)}."""
    rng = np.random.default_rng(seed)
    champion_counts: dict[str, int] = {}

    for _ in range(n_sims):
        qualifiers = []
        for group_name in sorted(groups.keys()):
            top2 = simulate_group(model, groups[group_name], rng=rng)
            qualifiers.extend(top2)

        champion = simulate_knockout(model, qualifiers, rng=rng)
        champion_counts[champion] = champion_counts.get(champion, 0) + 1

    return {
        t: c / n_sims
        for t, c in sorted(champion_counts.items(), key=lambda x: -x[1])
    }
