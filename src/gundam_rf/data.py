from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "death_in_war_target"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the Gundam character death dataset."""
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column is missing: {TARGET_COLUMN}")
    return df


def make_weak_binary_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    unknown_as_negative: bool = True,
) -> pd.Series:
    """Create a binary target.

    Current first-pass data has confirmed positives and Unknown rows.
    When unknown_as_negative=True, Unknown is converted to 0 as a weak baseline.

    This is not a factual survival label.
    """
    raw = df[target_column].astype(str).str.strip()

    y = pd.Series(index=df.index, dtype="float64")
    y.loc[raw.eq("1")] = 1

    if unknown_as_negative:
        y.loc[raw.str.lower().eq("unknown")] = 0
    else:
        y.loc[raw.str.lower().eq("unknown")] = pd.NA

    unknown_values = sorted(set(raw[y.isna()].dropna().tolist()))
    if unknown_values:
        raise ValueError(
            "Target contains unsupported values. "
            f"Supported values are '1' and 'Unknown'. Found: {unknown_values}"
        )

    return y.astype(int)


def summarize_labels(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Return a compact label summary table."""
    return (
        df[target_column]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(target_column)
        .reset_index(name="count")
    )
