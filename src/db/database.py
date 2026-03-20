"""SQLite database for Engine Health Agent.

Stores prediction history, agent run logs, drift metrics, and raw sensor data.

Tables:
    predictions    — per-engine RUL inference log
    agent_runs     — full agent execution history (replaces outputs/agent_runs.jsonl)
    drift_metrics  — PSI drift check snapshots
    raw_sensor_data — CMAPSS txt data imported once on first run
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]

# Resolve DB path from env, defaulting to data/engine_health.db
_DB_PATH: Optional[Path] = None


def _get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        env_path = os.getenv("DB_PATH", "data/engine_health.db")
        _DB_PATH = _ROOT / env_path if not Path(env_path).is_absolute() else Path(env_path)
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def get_db() -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist and run the raw-data migration."""
    conn = get_db()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                engine_id     INTEGER NOT NULL,
                rul           REAL    NOT NULL,
                lower_80      REAL,
                upper_80      REAL,
                std           REAL,
                current_cycle INTEGER
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                engine_id      INTEGER,
                query          TEXT,
                recommendation TEXT,
                tool_calls     TEXT,
                turns          INTEGER,
                latency_ms     INTEGER,
                error          TEXT
            );

            CREATE TABLE IF NOT EXISTS drift_metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                feature     TEXT    NOT NULL,
                condition   INTEGER,
                n_test      INTEGER,
                mean_test   REAL,
                std_test    REAL,
                mean_train  REAL,
                std_train   REAL,
                psi_score   REAL,
                drift_level TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_sensor_data (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT    NOT NULL,
                split   TEXT    NOT NULL,
                unit    INTEGER NOT NULL,
                cycle   INTEGER NOT NULL,
                op1 REAL, op2 REAL, op3 REAL,
                s1  REAL, s2  REAL, s3  REAL, s4  REAL, s5  REAL, s6  REAL, s7  REAL,
                s8  REAL, s9  REAL, s10 REAL, s11 REAL, s12 REAL, s13 REAL, s14 REAL,
                s15 REAL, s16 REAL, s17 REAL, s18 REAL, s19 REAL, s20 REAL, s21 REAL
            );

            CREATE INDEX IF NOT EXISTS idx_pred_engine ON predictions(engine_id);
            CREATE INDEX IF NOT EXISTS idx_runs_engine ON agent_runs(engine_id);
            CREATE INDEX IF NOT EXISTS idx_drift_ts    ON drift_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_raw_unit    ON raw_sensor_data(dataset, split, unit);
        """)
    conn.close()

    # Auto-migrate raw data on first run
    _auto_migrate()


def _auto_migrate() -> None:
    """Import raw CMAPSS txt files into raw_sensor_data if table is empty."""
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM raw_sensor_data").fetchone()[0]
    conn.close()
    if count > 0:
        return  # already imported

    for fd_id in ("FD004",):
        for split in ("train", "test"):
            txt = _ROOT / "data" / "raw" / f"{split}_{fd_id}.txt"
            if txt.exists():
                migrate_raw_to_db(fd_id, split)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def log_prediction(engine_id: int, pred: Dict[str, Any]) -> None:
    """Insert one RUL prediction into the predictions table."""
    try:
        conn = get_db()
        with conn:
            conn.execute(
                """INSERT INTO predictions
                   (timestamp, engine_id, rul, lower_80, upper_80, std, current_cycle)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    engine_id,
                    float(pred.get("rul", 0)),
                    float(pred.get("lower_80", 0)),
                    float(pred.get("upper_80", 0)),
                    float(pred.get("std", 0)),
                    int(pred.get("current_cycle", 0)),
                ),
            )
        conn.close()
    except Exception:
        pass  # never crash the main inference path


def log_agent_run(run: Dict[str, Any]) -> None:
    """Insert one agent run record into the agent_runs table."""
    try:
        conn = get_db()
        with conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (timestamp, engine_id, query, recommendation, tool_calls,
                    turns, latency_ms, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    run.get("engine_id"),
                    run.get("query"),
                    run.get("recommendation"),
                    json.dumps(run.get("tool_calls", [])),
                    run.get("turns"),
                    run.get("latency_ms"),
                    run.get("error"),
                ),
            )
        conn.close()
    except Exception:
        pass


def log_drift_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Bulk-insert drift metric rows."""
    if not metrics:
        return
    conn = get_db()
    ts = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.executemany(
            """INSERT INTO drift_metrics
               (timestamp, feature, condition, n_test, mean_test, std_test,
                mean_train, std_train, psi_score, drift_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ts,
                    m["feature"],
                    m.get("condition"),
                    m.get("n_test"),
                    m.get("mean_test"),
                    m.get("std_test"),
                    m.get("mean_train"),
                    m.get("std_train"),
                    m.get("psi_score"),
                    m.get("drift_level", "none"),
                )
                for m in metrics
            ],
        )
    conn.close()


def migrate_raw_to_db(fd_id: str, split: str) -> int:
    """Import a raw CMAPSS txt file into raw_sensor_data. Returns rows inserted."""
    import pandas as pd

    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    txt = _ROOT / "data" / "raw" / f"{split}_{fd_id}.txt"
    if not txt.exists():
        return 0

    df = pd.read_csv(txt, sep=r"\s+", header=None, names=cols)
    df.insert(0, "split", split)
    df.insert(0, "dataset", fd_id)

    conn = get_db()
    df.to_sql("raw_sensor_data", conn, if_exists="append", index=False)
    conn.commit()
    n = len(df)
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_predictions(engine_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
    """Return recent prediction rows as a list of dicts."""
    conn = get_db()
    if engine_id is not None:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE engine_id=? ORDER BY id DESC LIMIT ?",
            (engine_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_drift() -> List[Dict]:
    """Return the most recent complete drift snapshot (all features at one timestamp)."""
    conn = get_db()
    row = conn.execute(
        "SELECT timestamp FROM drift_metrics ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        conn.close()
        return []
    ts = row["timestamp"]
    rows = conn.execute(
        "SELECT * FROM drift_metrics WHERE timestamp=? ORDER BY psi_score DESC",
        (ts,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialise on import
init_db()
