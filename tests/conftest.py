"""Shared pytest fixtures for DataVaidya."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable so tests can `from core...import...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small, clean dataframe with no quality issues."""
    return pd.DataFrame({
        "id": range(1, 51),
        "age": np.random.RandomState(42).randint(18, 70, 50),
        "income": np.random.RandomState(42).normal(50000, 15000, 50).round(2),
        "city": np.random.RandomState(42).choice(["Mumbai", "Delhi", "Bangalore"], 50),
    })


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """A dataframe with seeded quality issues — missing, duplicates, outliers, mixed dtype."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "age": rng.randint(18, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n).round(2),
        "city": rng.choice(["Mumbai", "Delhi", "Bangalore", None], n, p=[.4, .3, .2, .1]),
        "constant_col": "fixed",
        "mixed": [str(i) if i % 7 else i for i in range(n)],
    })
    df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "income"] = 9_999_999  # outliers
    df = pd.concat([df, df.head(5)], ignore_index=True)  # 5 duplicates
    return df


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1], "b": ["x"]})
