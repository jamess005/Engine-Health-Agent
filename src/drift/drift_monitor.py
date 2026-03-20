"""Drift monitor — detects distribution shift between training and test sensor data.

Uses Population Stability Index (PSI) to compare each sensor feature's distribution
in the test fleet against the training baseline.

PSI thresholds:
    < 0.10  → no drift        (distribution unchanged)
    0.10–0.25 → moderate drift (worth monitoring)
    > 0.25  → significant drift (model may be degraded — investigate)

Usage::

    from src.drift.drift_monitor import run_drift_check
    summary = run_drift_check()  # computes PSI, stores to DB, returns summary

    from src.drift.drift_monitor import get_drift_summary
    summary = get_drift_summary()  # returns latest stored results (no recompute)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")

# PSI thresholds
_PSI_MODERATE = 0.10
_PSI_SIGNIFICANT = 0.25

# Sensor columns to monitor (informative sensors only — same set as anomaly detector)
_MONITOR_COLS = [
    "n_s2", "n_s3", "n_s4", "n_s7", "n_s8",
    "n_s11", "n_s13", "n_s14", "n_s15",
]

# Feature columns including operating-condition channels
_EXTRA_COLS = ["os1", "os2", "os3"]


def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions.

    Args:
        baseline: Training distribution samples.
        current:  Current (test) distribution samples.
        bins:     Number of histogram bins (default 10).

    Returns:
        PSI score (0 = identical, higher = more drift).
    """
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Shared bin edges from combined range
    combined = np.concatenate([baseline, current])
    edges = np.histogram_bin_edges(combined, bins=bins)

    exp_counts, _ = np.histogram(baseline, bins=edges)
    act_counts, _ = np.histogram(current, bins=edges)

    # Avoid division by zero and log(0) by flooring at a small value
    n_base = len(baseline)
    n_curr = len(current)
    exp_pct = np.clip(exp_counts / n_base, 1e-6, None)
    act_pct = np.clip(act_counts / n_curr, 1e-6, None)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 5)


def _drift_level(psi: float) -> str:
    if psi >= _PSI_SIGNIFICANT:
        return "significant"
    elif psi >= _PSI_MODERATE:
        return "moderate"
    return "none"


def run_drift_check(fd_id: str = "FD004") -> Dict[str, Any]:
    """Compute PSI for all monitored sensor features and store results to DB.

    Compares the test fleet distribution (data/processed/{fd_id}_test.parquet)
    against the training baseline (data/processed/{fd_id}_train.parquet).

    Args:
        fd_id: Dataset identifier (default 'FD004').

    Returns:
        summary dict with keys:
            features_checked, no_drift, moderate_drift, significant_drift,
            flagged (list of feature names with PSI ≥ 0.10),
            details (list of per-feature dicts)
    """
    from src.features.loader import load_processed
    from src.features.builder import build_features
    from src.db.database import log_drift_metrics

    train_raw = load_processed(fd_id, "train")
    test_raw  = load_processed(fd_id, "test")

    train_feat = build_features(train_raw, add_rolling=False)
    test_feat  = build_features(test_raw, train_ref=train_feat, add_rolling=False)

    cols_to_check = [c for c in _MONITOR_COLS + _EXTRA_COLS if c in train_feat.columns]

    rows: List[Dict] = []
    for col in cols_to_check:
        baseline = train_feat[col].dropna().values
        current  = test_feat[col].dropna().values

        psi = compute_psi(baseline, current)
        level = _drift_level(psi)

        rows.append({
            "feature":    col,
            "condition":  None,
            "n_test":     int(len(current)),
            "mean_test":  float(np.nanmean(current)),
            "std_test":   float(np.nanstd(current)),
            "mean_train": float(np.nanmean(baseline)),
            "std_train":  float(np.nanstd(baseline)),
            "psi_score":  psi,
            "drift_level": level,
        })

    log_drift_metrics(rows)

    flagged = [r["feature"] for r in rows if r["drift_level"] != "none"]
    counts = {"none": 0, "moderate": 0, "significant": 0}
    for r in rows:
        counts[r["drift_level"]] += 1

    return {
        "features_checked": len(rows),
        "no_drift":         counts["none"],
        "moderate_drift":   counts["moderate"],
        "significant_drift": counts["significant"],
        "flagged":          flagged,
        "details":          sorted(rows, key=lambda r: -r["psi_score"]),
    }


def get_drift_summary() -> Optional[Dict[str, Any]]:
    """Return the latest stored drift snapshot without recomputing.

    Returns None if no drift check has been run yet.
    """
    from src.db.database import get_latest_drift

    rows = get_latest_drift()
    if not rows:
        return None

    flagged = [r["feature"] for r in rows if r["drift_level"] != "none"]
    counts = {"none": 0, "moderate": 0, "significant": 0}
    for r in rows:
        counts[r["drift_level"]] += 1

    return {
        "timestamp":        rows[0]["timestamp"],
        "features_checked": len(rows),
        "no_drift":         counts["none"],
        "moderate_drift":   counts["moderate"],
        "significant_drift": counts["significant"],
        "flagged":          flagged,
        "details":          rows,
    }
