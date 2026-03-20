"""Condition normaliser — removes operating-condition (altitude) effect from sensors.

Wraps the fitted KMeans and per-cluster z-score parameters saved in
models/preprocessing/.  No retraining: all parameters are loaded at init.
"""

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_PREP = _ROOT / "models" / "preprocessing"

# Sensors that carry degradation signal (dead channels and duplicates already excluded)
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s11", "s13", "s14", "s15"]
OP_COLS = ["op1", "op2", "op3"]


class ConditionNormaliser:
    """Per-cluster z-score normaliser for turbofan sensor data.

    Fitted on FD004 training engines using KMeans(k=6) on operating settings.
    Each sensor reading is standardised within its assigned flight condition.
    """

    def __init__(self) -> None:
        self._kmeans = joblib.load(_PREP / "condition_kmeans.pkl")
        self._scaler = joblib.load(_PREP / "condition_scaler.pkl")
        self._cluster_stats: Dict = joblib.load(_PREP / "cluster_stats.pkl")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign condition cluster and z-score each sensor within its cluster.

        Args:
            df: DataFrame with columns [op1, op2, op3, s2, s3, ..., s15].

        Returns:
            Copy of df with added columns:
            - ``condition``: cluster label 0–5
            - ``n_s2`` … ``n_s15``: condition-normalised sensor readings
        """
        df = df.copy()
        op_scaled = self._scaler.transform(df[OP_COLS])
        df["condition"] = self._kmeans.predict(op_scaled)

        for s in SENSORS:
            normed = np.zeros(len(df), dtype=np.float32)
            for cond in range(6):
                mask = df["condition"] == cond
                if not mask.any():
                    continue
                mu = self._cluster_stats[cond][s]["mean"]
                sd = self._cluster_stats[cond][s]["std"]
                idx = np.asarray(mask.values, dtype=np.bool_)
                normed[idx] = (df.loc[mask, s].values - mu) / (sd + 1e-8)
            df[f"n_{s}"] = normed

        return df
