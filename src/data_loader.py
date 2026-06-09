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
        (df["tournament"] == tournament_name)
        & (df["date"].dt.year == year)
    )
    return df[mask].copy()


WORLD_CUP_DATES = {
    "WC2010": {"name": "FIFA World Cup", "year": 2010, "cutoff": "2010-06-11"},
    "WC2014": {"name": "FIFA World Cup", "year": 2014, "cutoff": "2014-06-12"},
    "WC2018": {"name": "FIFA World Cup", "year": 2018, "cutoff": "2018-06-14"},
    "WC2022": {"name": "FIFA World Cup", "year": 2022, "cutoff": "2022-11-20"},
}
