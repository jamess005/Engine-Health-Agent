"""SQLite database for Engine Health Agent.

Stores prediction history, agent run logs, drift metrics, raw sensor data,
processed features, ground-truth RUL labels, and async API job results.

Tables:
    predictions        — per-engine RUL inference log
    agent_runs         — full agent execution history
    drift_metrics      — PSI drift check snapshots
    raw_sensor_data    — CMAPSS raw txt data (unit/cycle/op1-3/s1-21)
    processed_features — condition-normalised features from processed parquet files
    rul_labels         — ground-truth RUL per test engine from RUL_FD004.txt
    async_jobs         — async API query results (replaces in-memory dict)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env before any os.getenv calls — safe no-op if file absent
load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]

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
    """Create all tables if they don't exist and run data migrations."""
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

            CREATE TABLE IF NOT EXISTS rul_labels (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset   TEXT    NOT NULL,
                engine_id INTEGER NOT NULL,
                rul_true  INTEGER NOT NULL,
                UNIQUE(dataset, engine_id)
            );

            CREATE TABLE IF NOT EXISTS async_jobs (
                job_id    TEXT PRIMARY KEY,
                status    TEXT NOT NULL,
                engine_id INTEGER,
                result    TEXT,
                created   TEXT NOT NULL,
                updated   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pred_engine ON predictions(engine_id);
            CREATE INDEX IF NOT EXISTS idx_runs_engine ON agent_runs(engine_id);
            CREATE INDEX IF NOT EXISTS idx_drift_ts    ON drift_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_raw_unit    ON raw_sensor_data(dataset, split, unit);
            CREATE INDEX IF NOT EXISTS idx_rul_dataset ON rul_labels(dataset, engine_id);
        """)
    conn.close()

    # processed_features schema is derived from the parquet file at migration time
    # (too many columns to hardcode; pandas to_sql creates it automatically)

    _auto_migrate()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _auto_migrate() -> None:
    """Run any outstanding data migrations (idempotent — checks row counts first)."""
    conn = get_db()

    raw_count  = conn.execute("SELECT COUNT(*) FROM raw_sensor_data").fetchone()[0]
    rul_count  = conn.execute("SELECT COUNT(*) FROM rul_labels").fetchone()[0]
    runs_count = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]

    feat_count = 0
    if _table_exists(conn, "processed_features"):
        feat_count = conn.execute("SELECT COUNT(*) FROM processed_features").fetchone()[0]

    conn.close()

    if raw_count == 0:
        for split in ("train", "test"):
            migrate_raw_to_db("FD004", split)

    if feat_count == 0:
        for split in ("train", "test"):
            migrate_processed_to_db("FD004", split)

    if rul_count == 0:
        migrate_rul_labels_to_db("FD004")

    if runs_count == 0:
        migrate_agent_runs_jsonl()


# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------

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


def migrate_processed_to_db(fd_id: str, split: str) -> int:
    """Import a processed parquet file into processed_features. Returns rows inserted.

    The table schema is created dynamically from the parquet columns on first call.
    """
    import pandas as pd

    path = _ROOT / "data" / "processed" / f"{fd_id}_{split}.parquet"
    if not path.exists():
        return 0

    df = pd.read_parquet(path)
    # Ensure string/categorical columns are plain str for SQLite compatibility
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype(str)
    # Parquet may already contain dataset/split; set/overwrite to ensure correctness
    df["dataset"] = fd_id
    df["split"] = split

    conn = get_db()
    df.to_sql(
        "processed_features", conn,
        if_exists="append", index=False,
        method="multi", chunksize=5000,
    )
    # Add index on first write (IF NOT EXISTS is safe on repeated calls)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feat_unit "
        "ON processed_features(dataset, split, unit)"
    )
    conn.commit()
    n = len(df)
    conn.close()
    return n


def migrate_rul_labels_to_db(fd_id: str) -> int:
    """Import ground-truth RUL labels from RUL_{fd_id}.txt into rul_labels.

    Engine IDs are 1-indexed to match the convention used throughout the codebase.
    Returns number of rows inserted.
    """
    import pandas as pd

    path = _ROOT / "data" / "raw" / f"RUL_{fd_id}.txt"
    if not path.exists():
        return 0

    series = pd.read_csv(path, header=None, names=["rul_true"])["rul_true"]
    conn = get_db()
    for i, rul in enumerate(series, start=1):
        try:
            conn.execute(
                "INSERT OR IGNORE INTO rul_labels (dataset, engine_id, rul_true) VALUES (?,?,?)",
                (fd_id, i, int(rul)),
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    n = len(series)
    conn.close()
    return n


def migrate_agent_runs_jsonl() -> int:
    """Import historical agent_runs.jsonl records into agent_runs table.

    Runs only if the table is empty and the jsonl file exists.
    Returns number of rows inserted.
    """
    jsonl_path = _ROOT / "outputs" / "agent_runs.jsonl"
    if not jsonl_path.exists():
        return 0

    records = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue

    if not records:
        return 0

    conn = get_db()
    for r in records:
        try:
            conn.execute(
                """INSERT INTO agent_runs
                   (timestamp, engine_id, query, recommendation, tool_calls,
                    turns, latency_ms, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    r.get("engine_id"),
                    r.get("query"),
                    r.get("recommendation"),
                    json.dumps(r.get("tool_calls", [])),
                    r.get("turns"),
                    r.get("latency_ms"),
                    r.get("error"),
                ),
            )
        except Exception:
            continue
    conn.commit()
    conn.close()
    return len(records)


# ---------------------------------------------------------------------------
# Write helpers (inference path — never raise)
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
        pass


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


def get_rul_labels(fd_id: str = "FD004") -> Dict[int, int]:
    """Return {engine_id: rul_true} mapping from rul_labels table."""
    conn = get_db()
    rows = conn.execute(
        "SELECT engine_id, rul_true FROM rul_labels WHERE dataset=? ORDER BY engine_id",
        (fd_id,),
    ).fetchall()
    conn.close()
    return {r["engine_id"]: r["rul_true"] for r in rows}


# ---------------------------------------------------------------------------
# Async job store (API job persistence)
# ---------------------------------------------------------------------------

def create_job(job_id: str, engine_id: Optional[int]) -> None:
    """Insert a new async job row with status='queued'."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        conn = get_db()
        with conn:
            conn.execute(
                "INSERT INTO async_jobs (job_id, status, engine_id, result, created, updated)"
                " VALUES (?, 'queued', ?, NULL, ?, ?)",
                (job_id, engine_id, ts, ts),
            )
        conn.close()
    except Exception:
        pass


def update_job(job_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> None:
    """Update status and optionally store the result JSON."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        conn = get_db()
        with conn:
            conn.execute(
                "UPDATE async_jobs SET status=?, result=?, updated=? WHERE job_id=?",
                (status, json.dumps(result) if result is not None else None, ts, job_id),
            )
        conn.close()
    except Exception:
        pass


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the async_jobs row for job_id, or None if not found."""
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT * FROM async_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        d = dict(row)
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except (json.JSONDecodeError, ValueError):
                pass
        return d
    except Exception:
        return None


# Initialise on import
init_db()
