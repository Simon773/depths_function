"""Shared helpers for loading and lightly flagging the SE weather station data."""

import os
from pathlib import Path

import pandas as pd

# project_root/data/SE_20160101.csv, project_root being the parent of this
# script's folder (e.g. test_eda/../data). Override with RAW_DATA_PATH if
# your data lives somewhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = os.environ.get(
    "RAW_DATA_PATH", str(PROJECT_ROOT / "data" / "SE_20160101.csv")
)

NUMERIC_COLS = ["dd", "ff", "precip", "hu", "td", "t", "psl"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d %H:%M")
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag physically implausible readings without dropping them."""
    df = df.copy()
    df["hu_out_of_range"] = df["hu"] > 100
    df["dd_out_of_range"] = (df["dd"] < 0) | (df["dd"] > 360)
    df["ff_negative"] = df["ff"] < 0
    return df


def station_equipment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station: which sensor families ever report a value."""
    g = df.groupby("number_sta")
    profile = pd.DataFrame(
        {
            "lat": g["lat"].first(),
            "lon": g["lon"].first(),
            "height_sta": g["height_sta"].first(),
            "n_timestamps": g.size(),
            "has_wind": g["dd"].apply(lambda s: s.notna().any()),
            "has_humidity": g["hu"].apply(lambda s: s.notna().any()),
            "has_temp": g["t"].apply(lambda s: s.notna().any()),
            "has_psl": g["psl"].apply(lambda s: s.notna().any()),
        }
    )
    return profile
