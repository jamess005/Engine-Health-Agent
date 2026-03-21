"""Drift monitor — detects distribution shift between training and test sensor data.

Uses Population Stability Index (PSI) to compare each sensor feature's distribution
in the test fleet against the training baseline.

PSI thresholds:
    < 0.10  → no drift        (distribution unchanged)
    0.10–0.25 → moderate drift (worth monitoring)
    > 0.25  → significant drift (model may be degraded — investigate)

Usage::

    from src.drift.drift_monitor import run_full_drift_check
    summary = run_full_drift_check()  # computes PSI for inputs + predictions, stores to DB

    from src.drift.drift_monitor import get_drift_summary
    summary = get_drift_summary()  # returns latest stored results (no recompute)
"""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.config import PSI_MODERATE as _PSI_MODERATE, PSI_SIGNIFICANT as _PSI_SIGNIFICANT

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

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
        logger.warning("PSI skipped — empty data after NaN/inf removal (baseline=%d, current=%d)",
                       len(baseline), len(current))
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


def run_drift_check(
    fd_id: str = "FD004",
    baseline_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute PSI for all monitored sensor features and store results to DB.

    Compares the test fleet distribution against the training baseline.

    Args:
        fd_id: Dataset identifier (default 'FD004').
        baseline_override: Optional DataFrame to use as the reference baseline
            instead of loading the training set.  Pass a recent window of
            known-good data to re-baseline the monitor without retraining.

    Returns:
        summary dict with keys:
            features_checked, no_drift, moderate_drift, significant_drift,
            flagged (list of feature names with PSI ≥ 0.10),
            details (list of per-feature dicts with drift_direction and mean_shift)
    """
    from src.features.loader import load_processed
    from src.features.builder import build_features
    from src.db.database import log_drift_metrics

    test_raw  = load_processed(fd_id, "test")
    test_feat = build_features(test_raw, add_rolling=False)

    if baseline_override is not None:
        train_feat = baseline_override
        logger.info("run_drift_check: using caller-supplied baseline (%d rows)", len(train_feat))
    else:
        train_raw  = load_processed(fd_id, "train")
        train_feat = build_features(train_raw, add_rolling=False)

    cols_to_check = [c for c in _MONITOR_COLS + _EXTRA_COLS if c in train_feat.columns]

    missing = set(_MONITOR_COLS + _EXTRA_COLS) - set(train_feat.columns)
    if missing:
        logger.warning(
            "Drift check: %d expected columns absent from features: %s",
            len(missing), sorted(missing),
        )

    rows: List[Dict] = []
    for col in cols_to_check:
        baseline = train_feat[col].dropna().values
        current  = test_feat[col].dropna().values if col in test_feat.columns else np.array([])

        psi = compute_psi(baseline, current)
        level = _drift_level(psi)

        mean_shift = float(np.nanmean(current) - np.nanmean(baseline)) if len(current) > 0 else 0.0
        if abs(mean_shift) < 0.05:
            drift_direction = "stable"
        elif mean_shift > 0:
            drift_direction = "up"
        else:
            drift_direction = "down"

        rows.append({
            "feature":         col,
            "condition":       None,
            "n_test":          int(len(current)),
            "mean_test":       float(np.nanmean(current)) if len(current) > 0 else 0.0,
            "std_test":        float(np.nanstd(current))  if len(current) > 0 else 0.0,
            "mean_train":      float(np.nanmean(baseline)),
            "std_train":       float(np.nanstd(baseline)),
            "psi_score":       psi,
            "drift_level":     level,
            "drift_direction": drift_direction,
            "mean_shift":      round(mean_shift, 4),
        })

    # Persist only the DB-schema fields (drift_direction/mean_shift are in-memory only)
    log_drift_metrics([{k: v for k, v in r.items()
                        if k not in ("drift_direction", "mean_shift")} for r in rows])

    flagged = [r["feature"] for r in rows if r["drift_level"] != "none"]
    counts = Counter(r["drift_level"] for r in rows)
    counts.setdefault("none", 0)
    counts.setdefault("moderate", 0)
    counts.setdefault("significant", 0)

    if counts["significant"] > 0:
        sig_features = ", ".join(r["feature"] for r in rows if r["drift_level"] == "significant")
        logger.warning(
            "SIGNIFICANT sensor drift detected in: %s — model accuracy may be degraded",
            sig_features,
        )
    elif counts["moderate"] > 0:
        logger.info("Moderate sensor drift detected in %d features — monitor closely",
                    counts["moderate"])
    else:
        logger.info("Sensor drift check complete — all %d features within normal range",
                    len(rows))

    return {
        "features_checked": len(rows),
        "no_drift":         counts["none"],
        "moderate_drift":   counts["moderate"],
        "significant_drift": counts["significant"],
        "flagged":          flagged,
        "details":          sorted(rows, key=lambda r: -r["psi_score"]),
    }


def run_output_drift_check(baseline_n: int = 50, recent_n: int = 50) -> Dict[str, Any]:
    """Monitor drift in the RUL prediction distribution.

    Compares the oldest ``baseline_n`` predictions in the DB against the most
    recent ``recent_n`` using PSI.  Returns an error dict if insufficient
    prediction history is available.

    Args:
        baseline_n: Number of oldest predictions to use as the reference.
        recent_n:   Number of most recent predictions to compare against.

    Returns:
        Dict with feature, psi_score, drift_level, n_baseline, n_recent,
        or an 'error' key if there is insufficient data.
    """
    from src.db.database import get_predictions, log_drift_metrics

    rows = get_predictions(limit=baseline_n + recent_n)
    if len(rows) < baseline_n + recent_n:
        msg = f"Need at least {baseline_n + recent_n} predictions; have {len(rows)}"
        logger.info("Output drift check skipped — %s", msg)
        return {"error": msg}

    # get_predictions returns newest-first (ORDER BY id DESC)
    recent_ruls   = np.array([r["rul"] for r in rows[:recent_n]])
    baseline_ruls = np.array([r["rul"] for r in rows[recent_n:]])

    psi = compute_psi(baseline_ruls, recent_ruls)
    level = _drift_level(psi)

    metric = {
        "feature":    "rul_predicted",
        "condition":  None,
        "n_test":     recent_n,
        "mean_test":  float(recent_ruls.mean()),
        "std_test":   float(recent_ruls.std()),
        "mean_train": float(baseline_ruls.mean()),
        "std_train":  float(baseline_ruls.std()),
        "psi_score":  psi,
        "drift_level": level,
    }
    log_drift_metrics([metric])

    if level == "significant":
        logger.warning(
            "SIGNIFICANT output drift detected (RUL predictions): PSI=%.4f — "
            "investigate data pipeline or model decay", psi,
        )
    elif level == "moderate":
        logger.info("Moderate output drift detected (RUL predictions): PSI=%.4f", psi)

    return {
        "feature":    "rul_predicted",
        "psi_score":  psi,
        "drift_level": level,
        "n_baseline": baseline_n,
        "n_recent":   recent_n,
    }


def run_full_drift_check(fd_id: str = "FD004") -> Dict[str, Any]:
    """Run both input (sensor) and output (prediction) drift checks.

    Args:
        fd_id: Dataset identifier (default 'FD004').

    Returns:
        Dict with 'input_drift' and 'output_drift' sections.
    """
    input_result  = run_drift_check(fd_id)
    output_result = run_output_drift_check()
    return {"input_drift": input_result, "output_drift": output_result}


def get_drift_summary() -> Optional[Dict[str, Any]]:
    """Return the latest stored drift snapshot without recomputing.

    Returns None if no drift check has been run yet.
    """
    from src.db.database import get_latest_drift

    rows = get_latest_drift()
    if not rows:
        return None

    flagged = [r["feature"] for r in rows if r["drift_level"] != "none"]
    counts = Counter(r["drift_level"] for r in rows)
    counts.setdefault("none", 0)
    counts.setdefault("moderate", 0)
    counts.setdefault("significant", 0)

    return {
        "timestamp":        rows[0]["timestamp"],
        "features_checked": len(rows),
        "no_drift":         counts["none"],
        "moderate_drift":   counts["moderate"],
        "significant_drift": counts["significant"],
        "flagged":          flagged,
        "details":          rows,
    }
