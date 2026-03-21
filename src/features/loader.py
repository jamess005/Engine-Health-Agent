"""Data loading utilities for NASA CMAPSS FD004.

All functions try the SQLite database first and fall back to the original
flat files (parquet / txt) if the database is unavailable or empty.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_RAW = _ROOT / "data" / "raw"
_PROCESSED = _ROOT / "data" / "processed"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_raw(fd_id: str, split: str = "train") -> pd.DataFrame:
    """Load raw CMAPSS sensor data.

    Tries raw_sensor_data table in SQLite first; falls back to the original
    txt file if the table is empty or the DB is unavailable.

    Args:
        fd_id: Dataset ID string, e.g. 'FD004'.
        split: 'train' or 'test'.

    Returns:
        DataFrame with columns [unit, cycle, op1, op2, op3, s1..s21, dataset].
    """
    try:
        from src.db.database import get_db
        conn = get_db()
        n = conn.execute(
            "SELECT COUNT(*) FROM raw_sensor_data WHERE dataset=? AND split=?",
            (fd_id, split),
        ).fetchone()[0]
        if n > 0:
            df = pd.read_sql(
                "SELECT * FROM raw_sensor_data WHERE dataset=? AND split=? ORDER BY unit, cycle",
                conn, params=(fd_id, split),
            ).drop(columns=["id", "split"], errors="ignore")
            conn.close()
            logger.debug("load_raw: loaded %s/%s from SQLite (%d rows)", fd_id, split, len(df))
            return df
        conn.close()
    except Exception as exc:
        logger.warning("load_raw: DB query failed for %s/%s, falling back to txt: %s", fd_id, split, exc)

    # Fallback: read from txt file
    path = _RAW / f"{split}_{fd_id}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["dataset"] = fd_id
    logger.info("load_raw: loaded %s/%s from txt file (DB unavailable or empty)", fd_id, split)
    return df


def load_processed(fd_id: str, split: str = "train", add_op_settings: bool = True) -> pd.DataFrame:
    """Load condition-normalised features.

    Tries processed_features table in SQLite first; falls back to the original
    parquet file if the table is empty or the DB is unavailable.

    Args:
        fd_id: Dataset ID string, e.g. 'FD004'.
        split: 'train' or 'test'.
        add_op_settings: If True, ensure op1/op2/op3 columns are present so that
            downstream feature builders can produce os1/os2/os3 contextual
            channels required by ForecastNet. (os1/os2/os3 stored in DB map
            directly to op1/op2/op3 — same values, different names.)

    Returns:
        DataFrame with condition-normalised features, RUL labels, and
        optionally op1/op2/op3 columns.
    """
    df = None

    try:
        from src.db.database import get_db
        conn = get_db()
        n = conn.execute(
            "SELECT COUNT(*) FROM processed_features WHERE dataset=? AND split=?",
            (fd_id, split),
        ).fetchone()[0]
        if n > 0:
            df = pd.read_sql(
                "SELECT * FROM processed_features "
                "WHERE dataset=? AND split=? ORDER BY unit, cycle",
                conn, params=(fd_id, split),
            ).drop(columns=["id", "dataset", "split"], errors="ignore")
            conn.close()
            logger.debug("load_processed: loaded %s/%s from SQLite (%d rows)", fd_id, split, len(df))

            # If op1/op2/op3 not present but os1/os2/os3 are, alias them back
            # (os1 = op1, os2 = op2, os3 = op3 — same values, builder needs op* name)
            if add_op_settings and "op1" not in df.columns:
                if "os1" in df.columns:
                    df["op1"] = df["os1"]
                    df["op2"] = df["os2"]
                    df["op3"] = df["os3"]
                else:
                    # Last resort: query raw_sensor_data for op settings
                    try:
                        raw_conn = get_db()
                        op_df = pd.read_sql(
                            "SELECT unit, cycle, op1, op2, op3 FROM raw_sensor_data "
                            "WHERE dataset=? AND split=?",
                            raw_conn, params=(fd_id, split),
                        )
                        raw_conn.close()
                        df = df.merge(op_df, on=["unit", "cycle"], how="left")
                    except Exception as exc:
                        logger.warning("load_processed: could not fetch op settings from raw_sensor_data: %s", exc)
            return df
        conn.close()
    except Exception as exc:
        logger.warning("load_processed: DB query failed for %s/%s, falling back to parquet: %s", fd_id, split, exc)

    # Fallback: read from parquet
    logger.info("load_processed: loading %s/%s from parquet (DB unavailable or empty)", fd_id, split)
    path = _PROCESSED / f"{fd_id}_{split}.parquet"
    df = pd.read_parquet(path)
    if add_op_settings and "op1" not in df.columns:
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

    Tries rul_labels table in SQLite first; falls back to the original txt
    file if unavailable.

    Returns:
        Series indexed 0..N-1 (one value per test engine, in order).
    """
    try:
        from src.db.database import get_db
        conn = get_db()
        rows = conn.execute(
            "SELECT rul_true FROM rul_labels WHERE dataset=? ORDER BY engine_id",
            (fd_id,),
        ).fetchall()
        conn.close()
        if rows:
            logger.debug("load_rul_labels: loaded %s from SQLite (%d engines)", fd_id, len(rows))
            return pd.Series([r[0] for r in rows], name="rul_true")
        conn.close()
    except Exception as exc:
        logger.warning("load_rul_labels: DB query failed for %s, falling back to txt: %s", fd_id, exc)

    # Fallback: read from txt file
    logger.info("load_rul_labels: loading %s from txt file (DB unavailable or empty)", fd_id)
    path = _RAW / f"RUL_{fd_id}.txt"
    return pd.read_csv(path, header=None, names=["rul_true"])["rul_true"]
