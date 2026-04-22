import logging
import random
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd


# Logger
def setup_logger(name: str = "jitai", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# Time utilities
def make_study_start(year: int = 2026, month: int = 2, day: int = 2) -> datetime:
    return datetime(year, month, day, 7, 0, 0)


def get_day_time(hour: int) -> str:
    if 5 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 20:
        return "Evening"
    else:
        return "Night"


def get_day_name(dt: datetime) -> str:
    return dt.strftime("%A")


def build_timestamp(start_dt: datetime, study_day: int, hour: int) -> str:
    base = start_dt.replace(hour=0, minute=0, second=0)
    dt   = base + timedelta(days=study_day, hours=hour)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + "+00:00"


def hours_in_day() -> List[int]:
    return list(range(7, 22))


# Participant identifiers
def generate_participant_ids(n: int, start: int = 6) -> List[str]:
    return [f"P{i}" for i in range(start, start + n)]

# Missing-data helper
def should_miss_notification(rng: np.random.Generator, miss_rate: float = 0.25) -> bool:
    return bool(rng.random() < miss_rate)

# Reproducibility

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

# Summary printer
def print_summary(df: pd.DataFrame) -> None:
    """Print a concise dataset summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  JITAI SIMULATION — DATASET SUMMARY")
    print(sep)

    total   = len(df)
    filled  = (df["Status"] == "yes").sum()
    partial = (df["Status"] == "yes_but_not_feasible").sum()
    missed  = (df["Status"] == "no").sum()
    truly_missing = df["Status"].isna().sum()
    obs     = df[df["Reward"].notna()]

    print(f"\n  Total rows                : {total:,}")
    print(f"  Status = yes              : {filled:,}  ({100*filled/total:.1f}%)")
    print(f"  Status = yes_but_not_feas : {partial:,}  ({100*partial/total:.1f}%)")
    print(f"  Status = no               : {missed:,}  ({100*missed/total:.1f}%)")
    print(f"  Truly missing slots       : {truly_missing:,}  ({100*truly_missing/total:.1f}%)")
    print(f"\n  Mean Reward               : {obs['Reward'].mean():.4f}")
    print(f"  Reward Std                : {obs['Reward'].std():.4f}")
    print(f"  Participants              : {df['Participant_Id'].nunique()}")

    print(f"\n  Context distribution (top 5):")
    ctx = (
        obs.groupby("Context")["Reward"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .head(5)
    )
    for ctx_name, row in ctx.iterrows():
        print(f"    {ctx_name:<22} n={int(row['count']):>5}  mean_reward={row['mean']:.4f}")

    print(f"\n  Mean Reward by Intervention (top 10):")
    top = (
        obs.groupby("Intervention")["Reward"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    for name, r in top.items():
        print(f"    {name:<45}  {r:.4f}")

    print(f"\n{sep}\n")