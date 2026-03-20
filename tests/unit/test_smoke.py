"""Smoke tests — fast checks for environment and DB schema.

Data-dependent tests (tables populated, loader returns rows) are skipped in CI
because the FD004 data files are gitignored.  They run locally once the DB has
been initialised by running any CLI command (which triggers _auto_migrate()).
"""

import pytest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DATA_PRESENT = (
    (_ROOT / "data" / "processed" / "FD004_test.parquet").exists()
    or (_ROOT / "data" / "raw" / "test_FD004.txt").exists()
)

_needs_data = pytest.mark.skipif(
    not _DATA_PRESENT,
    reason="FD004 data files not available (gitignored — run locally after data setup)",
)


def test_environment():
    import pandas as pd  # noqa: F401
    import numpy as np  # noqa: F401
    import xgboost as xgb  # noqa: F401
    import sklearn  # noqa: F401
    assert True


def test_db_schema():
    """Verify the SQLite database initialises with the core required tables."""
    from src.db.database import get_db
    conn = get_db()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    # These tables are always created by init_db() — no data files needed
    required_always = {"predictions", "agent_runs", "drift_metrics",
                       "raw_sensor_data", "rul_labels", "async_jobs"}
    assert required_always.issubset(tables), f"Missing tables: {required_always - tables}"


@_needs_data
def test_db_tables():
    """Verify the DB has all tables including processed_features (requires parquet)."""
    from src.db.database import get_db
    conn = get_db()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    required = {"predictions", "agent_runs", "drift_metrics",
                "raw_sensor_data", "processed_features", "rul_labels"}
    assert required.issubset(tables), f"Missing tables: {required - tables}"


@_needs_data
def test_db_populated():
    """Verify key tables have been populated with FD004 data."""
    from src.db.database import get_db
    conn = get_db()
    raw_n  = conn.execute("SELECT COUNT(*) FROM raw_sensor_data").fetchone()[0]
    feat_n = conn.execute("SELECT COUNT(*) FROM processed_features").fetchone()[0]
    rul_n  = conn.execute("SELECT COUNT(*) FROM rul_labels").fetchone()[0]
    conn.close()
    assert raw_n  > 0, "raw_sensor_data is empty"
    assert feat_n > 0, "processed_features is empty"
    assert rul_n  == 248, f"rul_labels should have 248 rows, got {rul_n}"


@_needs_data
def test_loader_db_first():
    """Verify load_processed() and load_rul_labels() return data from the DB."""
    from src.features.loader import load_processed, load_rul_labels
    df = load_processed("FD004", "test")
    assert len(df) > 0, "load_processed returned empty DataFrame"
    assert "unit" in df.columns and "cycle" in df.columns

    labels = load_rul_labels("FD004")
    assert len(labels) == 248, f"Expected 248 RUL labels, got {len(labels)}"
    assert labels.min() > 0
