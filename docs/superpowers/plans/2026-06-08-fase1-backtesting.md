# Fase 1 - Backtesting & Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a walk-forward backtesting harness that measures the v0 static Poisson model on WC2010/2014/2018/2022, producing RPS/log-loss/Brier/ECE metrics and calibration curves.

**Architecture:** Data loader fetches martj42 results.csv. Static strength model fits Poisson regression with temporal decay. Backtest harness enforces strict walk-forward (no leakage). Metrics module computes RPS, log-loss, Brier, ECE. Simulation module runs Monte Carlo tournament brackets.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib

---

## File Structure

```
worldcup26/
├── data/
│   └── results.csv                 # Downloaded from martj42
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load + filter results.csv
│   ├── strength_static.py          # v0 Poisson + Ridge baseline
│   ├── metrics.py                  # RPS, log-loss, Brier, ECE
│   ├── backtest.py                 # Walk-forward harness
│   └── simulate.py                 # Monte Carlo tournament sim
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   ├── test_no_leak.py
│   └── test_strength_static.py
├── notebooks/
│   └── calibration.ipynb
├── requirements.txt
├── SPEC_v1.md
└── .gitignore
```

---

### Task 1: Project scaffolding & data download

**Files:**
- Create: `requirements.txt`, `.gitignore`, `src/__init__.py`, `tests/__init__.py`
- Download: `data/results.csv`

- [ ] **Step 1: Create requirements.txt**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
statsmodels>=0.14
matplotlib>=3.7
pytest>=7.4
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.ipynb_checkpoints/
```

- [ ] **Step 3: Create src/__init__.py and tests/__init__.py** (empty files)

- [ ] **Step 4: Download results.csv**

```bash
curl -L -o data/results.csv "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
```

Verify: file should have columns `date,home_team,away_team,home_score,away_score,tournament,city,country,neutral`

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore src/__init__.py tests/__init__.py data/results.csv SPEC_v1.md
git commit -m "chore: project scaffolding + results.csv dataset"
```

---

### Task 2: Data loader

**Files:**
- Create: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data_loader.py
import pandas as pd
from src.data_loader import load_results, filter_before

def test_load_results_columns():
    df = load_results()
    assert "date" in df.columns
    assert "home_team" in df.columns
    assert "away_team" in df.columns
    assert "home_score" in df.columns
    assert "away_score" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

def test_load_results_not_empty():
    df = load_results()
    assert len(df) > 40000

def test_filter_before():
    df = load_results()
    cutoff = pd.Timestamp("2010-06-01")
    filtered = filter_before(df, cutoff)
    assert filtered["date"].max() < cutoff

def test_filter_before_preserves_data():
    df = load_results()
    cutoff = pd.Timestamp("2020-01-01")
    filtered = filter_before(df, cutoff)
    assert len(filtered) > 0
    assert len(filtered) < len(df)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_data_loader.py -v
```

- [ ] **Step 3: Implement data_loader.py**

```python
# src/data_loader.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "results.csv"

def load_results(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def filter_before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return df[df["date"] < cutoff].copy()

def get_tournament_matches(df: pd.DataFrame, tournament_name: str, year: int) -> pd.DataFrame:
    mask = (
        df["tournament"].str.contains(tournament_name, case=False, na=False)
        & (df["date"].dt.year == year)
    )
    return df[mask].copy()

WORLD_CUP_DATES = {
    "WC2010": {"name": "FIFA World Cup", "year": 2010, "cutoff": "2010-06-11"},
    "WC2014": {"name": "FIFA World Cup", "year": 2014, "cutoff": "2014-06-12"},
    "WC2018": {"name": "FIFA World Cup", "year": 2018, "cutoff": "2018-06-14"},
    "WC2022": {"name": "FIFA World Cup", "year": 2022, "cutoff": "2022-11-20"},
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_data_loader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader with filtering and tournament extraction"
```

---

### Task 3: Metrics module (RPS, log-loss, Brier, ECE)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
from src.metrics import rps, log_loss_1x2, brier_multi, calibration_ece

def test_rps_perfect_prediction():
    # Perfect prediction: outcome is home win (idx 0), predicted [1, 0, 0]
    score = rps(np.array([1.0, 0.0, 0.0]), 0)
    assert score == 0.0

def test_rps_worst_prediction():
    # Worst: outcome is away win (idx 2), predicted [1, 0, 0]
    score = rps(np.array([1.0, 0.0, 0.0]), 2)
    assert score == 1.0

def test_rps_uniform():
    # Uniform prediction [1/3, 1/3, 1/3], outcome home win
    score = rps(np.array([1/3, 1/3, 1/3]), 0)
    expected = 0.5 * ((1/3 - 1)**2 + (2/3 - 1)**2)
    assert abs(score - expected) < 1e-10

def test_rps_symmetric():
    # RPS is ordered: predicting away when home happened != predicting draw
    s1 = rps(np.array([0.0, 0.0, 1.0]), 0)  # predicted away, was home
    s2 = rps(np.array([0.0, 1.0, 0.0]), 0)  # predicted draw, was home
    assert s1 > s2  # farther away in ordering = worse

def test_log_loss_perfect():
    probs = np.array([[0.99, 0.005, 0.005]])
    outcomes = np.array([0])
    score = log_loss_1x2(probs, outcomes)
    assert score < 0.02

def test_brier_perfect():
    probs = np.array([[1.0, 0.0, 0.0]])
    outcomes = np.array([0])
    score = brier_multi(probs, outcomes)
    assert score == 0.0

def test_calibration_ece_perfect():
    # If predicted probabilities match observed frequencies, ECE ~ 0
    n = 1000
    probs = np.column_stack([
        np.full(n, 0.5),
        np.full(n, 0.3),
        np.full(n, 0.2),
    ])
    rng = np.random.default_rng(42)
    outcomes = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    ece = calibration_ece(probs, outcomes, n_bins=5)
    assert ece < 0.1  # should be close to 0 with enough samples
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement metrics.py**

```python
# src/metrics.py
import numpy as np

def rps(probs: np.ndarray, outcome_idx: int) -> float:
    """Ranked Probability Score for ordered 3-way outcome (H/D/A)."""
    cumulative_pred = np.cumsum(probs)
    cumulative_real = np.cumsum(np.eye(3)[outcome_idx])
    return float(0.5 * np.sum((cumulative_pred - cumulative_real) ** 2))

def log_loss_1x2(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Mean log-loss over match predictions. probs shape (N, 3), outcomes shape (N,)."""
    probs = np.clip(probs, eps, 1 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n = len(outcomes)
    return float(-np.sum(np.log(probs[np.arange(n), outcomes])) / n)

def brier_multi(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean multiclass Brier score. probs (N, 3), outcomes (N,)."""
    one_hot = np.eye(3)[outcomes]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def calibration_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error across all 3 classes."""
    total = 0.0
    n = len(outcomes)
    for cls in range(3):
        p = probs[:, cls]
        y = (outcomes == cls).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            avg_pred = p[mask].mean()
            avg_true = y[mask].mean()
            total += mask.sum() * abs(avg_pred - avg_true)
    return float(total / (n * 3))

def match_outcome(home_score: int, away_score: int) -> int:
    """Return 0=home win, 1=draw, 2=away win."""
    if home_score > away_score:
        return 0
    elif home_score == away_score:
        return 1
    else:
        return 2
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: metrics module (RPS, log-loss, Brier, ECE)"
```

---

### Task 4: Static strength model (v0 baseline)

**Files:**
- Create: `src/strength_static.py`
- Test: `tests/test_strength_static.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strength_static.py
import pandas as pd
import numpy as np
from src.data_loader import load_results, filter_before
from src.strength_static import train_model, predict_match

def test_train_model_returns_model():
    df = load_results()
    cutoff = pd.Timestamp("2010-06-01")
    model = train_model(df, cutoff)
    assert model is not None
    assert hasattr(model, "predict_match")

def test_predict_match_probabilities():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_model(df, cutoff)
    probs = model.predict_match("Brazil", "Germany", neutral=True)
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(p >= 0 for p in probs)

def test_predict_home_advantage():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_model(df, cutoff)
    probs_home = model.predict_match("Brazil", "Bolivia", neutral=False)
    probs_neutral = model.predict_match("Brazil", "Bolivia", neutral=True)
    # Home advantage should give Brazil higher win prob
    assert probs_home[0] > probs_neutral[0]

def test_strong_team_beats_weak():
    df = load_results()
    cutoff = pd.Timestamp("2018-06-01")
    model = train_model(df, cutoff)
    probs = model.predict_match("Brazil", "Luxembourg", neutral=True)
    assert probs[0] > 0.5  # Brazil should be favored
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_strength_static.py -v
```

- [ ] **Step 3: Implement strength_static.py**

The v0 model: Poisson regression for goals scored with team attack/defense effects,
temporal decay (half-life 2 years), then Monte Carlo over scorelines for 1-X-2 probabilities.

```python
# src/strength_static.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

class StaticModel:
    def __init__(self, teams, attack, defense, home_adv, intercept):
        self.teams = teams
        self.team_to_idx = {t: i for i, t in enumerate(teams)}
        self.attack = attack
        self.defense = defense
        self.home_adv = home_adv
        self.intercept = intercept

    def predict_match(self, home: str, away: str, neutral: bool = False) -> tuple:
        """Return (p_home_win, p_draw, p_away_win)."""
        hi = self.team_to_idx.get(home)
        ai = self.team_to_idx.get(away)
        if hi is None or ai is None:
            return (1/3, 1/3, 1/3)

        home_lambda = np.exp(
            self.intercept + self.attack[hi] - self.defense[ai]
            + (self.home_adv if not neutral else 0)
        )
        away_lambda = np.exp(
            self.intercept + self.attack[ai] - self.defense[hi]
        )

        home_lambda = np.clip(home_lambda, 0.1, 8.0)
        away_lambda = np.clip(away_lambda, 0.1, 8.0)

        max_goals = 10
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for h in range(max_goals):
            for a in range(max_goals):
                p = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)
                if h > a:
                    p_home += p
                elif h == a:
                    p_draw += p
                else:
                    p_away += p

        total = p_home + p_draw + p_away
        return (p_home / total, p_draw / total, p_away / total)


def train_model(df: pd.DataFrame, cutoff: pd.Timestamp, half_life_years: float = 2.0) -> StaticModel:
    """Train Poisson model on matches before cutoff with temporal decay."""
    train = df[df["date"] < cutoff].copy()
    train = train.dropna(subset=["home_score", "away_score"])
    train["home_score"] = train["home_score"].astype(int)
    train["away_score"] = train["away_score"].astype(int)

    # Temporal weights
    days_diff = (cutoff - train["date"]).dt.days.values
    half_life_days = half_life_years * 365.25
    weights = np.exp(-np.log(2) * days_diff / half_life_days)

    # Get teams with enough matches (weighted)
    all_teams = pd.concat([train["home_team"], train["away_team"]]).unique()
    teams = sorted(all_teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Build arrays
    home_idx = train["home_team"].map(team_to_idx).values
    away_idx = train["away_team"].map(team_to_idx).values
    home_goals = train["home_score"].values.astype(float)
    away_goals = train["away_score"].values.astype(float)
    is_neutral = train["neutral"].fillna(False).values.astype(float)

    # Poisson log-likelihood with L2 regularization
    reg_lambda = 0.01

    def neg_log_lik(params):
        intercept = params[0]
        home_adv = params[1]
        attack = params[2:2 + n_teams]
        defense = params[2 + n_teams:2 + 2 * n_teams]

        # Home team expected goals
        mu_h = np.exp(intercept + attack[home_idx] - defense[away_idx] + home_adv * (1 - is_neutral))
        # Away team expected goals
        mu_a = np.exp(intercept + attack[away_idx] - defense[home_idx])

        mu_h = np.clip(mu_h, 1e-6, 20)
        mu_a = np.clip(mu_a, 1e-6, 20)

        ll_h = weights * (home_goals * np.log(mu_h) - mu_h)
        ll_a = weights * (away_goals * np.log(mu_a) - mu_a)

        reg = reg_lambda * (np.sum(attack**2) + np.sum(defense**2))

        return -(np.sum(ll_h) + np.sum(ll_a)) + reg

    # Initial params
    x0 = np.zeros(2 + 2 * n_teams)
    x0[0] = 0.3  # intercept ~ log(avg goals)

    result = minimize(neg_log_lik, x0, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-8})

    params = result.x
    intercept = params[0]
    home_adv = params[1]
    attack = params[2:2 + n_teams]
    defense = params[2 + n_teams:2 + 2 * n_teams]

    # Zero-sum constraint
    attack -= attack.mean()
    defense -= defense.mean()

    return StaticModel(teams, attack, defense, home_adv, intercept)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_strength_static.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/strength_static.py tests/test_strength_static.py
git commit -m "feat: static Poisson strength model (v0 baseline)"
```

---

### Task 5: Backtest harness with no-leak enforcement

**Files:**
- Create: `src/backtest.py`
- Test: `tests/test_no_leak.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_no_leak.py
import pandas as pd
from src.data_loader import load_results, WORLD_CUP_DATES
from src.backtest import backtest

def test_no_leakage():
    """Verify that for every prediction, train data ends before test data starts."""
    results = backtest(["WC2018"])
    assert len(results) > 0
    for _, row in results.iterrows():
        assert row["train_cutoff"] <= row["match_date"], (
            f"LEAK: train_cutoff {row['train_cutoff']} > match_date {row['match_date']}"
        )

def test_backtest_has_required_columns():
    results = backtest(["WC2018"])
    required = ["match_date", "home_team", "away_team", "home_score", "away_score",
                 "outcome", "p_home", "p_draw", "p_away", "train_cutoff", "tournament_id"]
    for col in required:
        assert col in results.columns, f"Missing column: {col}"

def test_backtest_probabilities_valid():
    results = backtest(["WC2018"])
    for _, row in results.iterrows():
        total = row["p_home"] + row["p_draw"] + row["p_away"]
        assert abs(total - 1.0) < 1e-4, f"Probs don't sum to 1: {total}"
        assert row["p_home"] >= 0
        assert row["p_draw"] >= 0
        assert row["p_away"] >= 0

def test_backtest_multiple_tournaments():
    results = backtest(["WC2014", "WC2018"])
    tournaments = results["tournament_id"].unique()
    assert len(tournaments) == 2
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/test_no_leak.py -v
```

- [ ] **Step 3: Implement backtest.py**

```python
# src/backtest.py
import pandas as pd
import numpy as np
from src.data_loader import load_results, filter_before, get_tournament_matches, WORLD_CUP_DATES
from src.strength_static import train_model
from src.metrics import match_outcome, rps, log_loss_1x2, brier_multi, calibration_ece

def backtest(tournament_ids: list[str], df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Walk-forward backtest over specified tournaments. Returns one row per match."""
    if df is None:
        df = load_results()

    rows = []
    for tid in tournament_ids:
        info = WORLD_CUP_DATES[tid]
        cutoff = pd.Timestamp(info["cutoff"])
        model = train_model(df, cutoff)
        matches = get_tournament_matches(df, info["name"], info["year"])

        for _, m in matches.iterrows():
            is_neutral = bool(m.get("neutral", True))
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

    # Per-tournament breakdown
    for tid in bt["tournament_id"].unique():
        mask = bt["tournament_id"] == tid
        t_probs = probs[mask]
        t_outcomes = outcomes[mask]
        t_rps = [rps(t_probs[i], t_outcomes[i]) for i in range(mask.sum())]
        report[f"{tid}_rps"] = float(np.mean(t_rps))
        report[f"{tid}_log_loss"] = log_loss_1x2(t_probs, t_outcomes)
        report[f"{tid}_n"] = int(mask.sum())

    return report
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_no_leak.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/backtest.py tests/test_no_leak.py
git commit -m "feat: walk-forward backtest harness with no-leak enforcement"
```

---

### Task 6: Tournament simulation (Monte Carlo)

**Files:**
- Create: `src/simulate.py`

- [ ] **Step 1: Implement simulate.py**

Monte Carlo simulation of a World Cup bracket (groups + knockout). Used for tournament-level metrics (P(champion)).

```python
# src/simulate.py
import numpy as np
from src.strength_static import StaticModel

def simulate_match(model: StaticModel, home: str, away: str, neutral: bool = True, rng=None):
    """Simulate a single match result. Returns (home_goals, away_goals)."""
    if rng is None:
        rng = np.random.default_rng()
    probs = model.predict_match(home, away, neutral=neutral)
    outcome = rng.choice([0, 1, 2], p=probs)
    # For knockout: if draw, pick winner by coin flip weighted by H/A strength
    return outcome  # 0=home, 1=draw, 2=away


def simulate_group(model: StaticModel, teams: list[str], rng=None) -> list[str]:
    """Simulate round-robin group, return top 2 teams by points then goal-based tiebreak."""
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

    # Sort by points, tiebreak by random
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


def simulate_tournament(model: StaticModel, groups: dict[str, list[str]],
                        n_sims: int = 10000, seed: int = 42) -> dict[str, float]:
    """Monte Carlo tournament simulation. Returns {team: P(champion)}."""
    rng = np.random.default_rng(seed)
    champion_counts = {}

    for _ in range(n_sims):
        # Group stage
        qualifiers = []
        for group_name in sorted(groups.keys()):
            top2 = simulate_group(model, groups[group_name], rng=rng)
            qualifiers.extend(top2)

        # Knockout (simplified: just bracket the qualifiers in order)
        champion = simulate_knockout(model, qualifiers, rng=rng)
        champion_counts[champion] = champion_counts.get(champion, 0) + 1

    return {t: c / n_sims for t, c in sorted(champion_counts.items(), key=lambda x: -x[1])}
```

- [ ] **Step 2: Commit**

```bash
git add src/simulate.py
git commit -m "feat: Monte Carlo tournament simulation (groups + knockout)"
```

---

### Task 7: Calibration notebook & main runner

**Files:**
- Create: `run_backtest.py` (main entry point)

- [ ] **Step 1: Create run_backtest.py**

```python
# run_backtest.py
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
    print(f"Calibration plot saved to {output_path}")


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
        print(f"  {tid}: RPS={report[f'{tid}_rps']:.4f}  LogLoss={report[f'{tid}_log_loss']:.4f}  N={report[f'{tid}_n']}")

    plot_calibration(bt)

    # Save detailed results
    bt.to_csv("backtest_results.csv", index=False)
    print(f"\nDetailed results saved to backtest_results.csv")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full backtest**

```bash
python run_backtest.py
```

- [ ] **Step 3: Commit**

```bash
git add run_backtest.py
git commit -m "feat: main backtest runner with calibration plots"
```

---

### Task 8: Run all tests & final verification

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

- [ ] **Step 2: Run backtest and verify output**

```bash
python run_backtest.py
```

Expected: Table with RPS/log-loss/Brier/ECE for WC2010-2022, plus calibration.png

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: Phase 1 complete - backtesting harness with calibration"
```
