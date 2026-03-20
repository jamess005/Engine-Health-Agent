"""Data loading utilities for NASA CMAPSS FD004."""

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "raw"
_PROCESSED = _ROOT / "data" / "processed"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_raw(fd_id: str, split: str = "train") -> pd.DataFrame:
    """Load a raw CMAPSS txt file.

    Args:
        fd_id: Dataset ID string, e.g. 'FD004'.
        split: 'train' or 'test'.

    Returns:
        DataFrame with columns [unit, cycle, op1, op2, op3, s1..s21].
    """
    path = _RAW / f"{split}_{fd_id}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["dataset"] = fd_id
    return df


def load_processed(fd_id: str, split: str = "train", add_op_settings: bool = True) -> pd.DataFrame:
    """Load a pre-processed parquet file.

    Args:
        fd_id: Dataset ID string, e.g. 'FD004'.
        split: 'train' or 'test'.
        add_op_settings: If True, merge op1/op2/op3 from the raw file so that
            downstream feature builders can produce os1/os2/os3 contextual
            channels required by ForecastNet.

    Returns:
        DataFrame with condition-normalised features, RUL labels, and
        optionally op1/op2/op3 columns.
    """
    path = _PROCESSED / f"{fd_id}_{split}.parquet"
    df = pd.read_parquet(path)
    if add_op_settings:
        raw = pd.read_csv(
            _RAW / f"{split}_{fd_id}.txt",
            sep=r"\s+",
            header=None,
            names=COLS,
            usecols=["unit", "cycle", "op1", "op2", "op3"],
        )
        df = df.merge(raw[["unit", "cycle", "op1", "op2", "op3"]],
                      on=["unit", "cycle"], how="left")
    return df


def load_rul_labels(fd_id: str) -> pd.Series:
    """Load ground-truth RUL values for the test set.

    Returns a Series indexed 0..N-1 (one value per test engine, in order).
    """
    path = _RAW / f"RUL_{fd_id}.txt"
    return pd.read_csv(path, header=None, names=["rul_true"])["rul_true"]
