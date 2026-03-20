"""Sliding window construction for ForecastNet inference.

Produces (N, window_size, 56) float32 arrays ready for the CNN.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SCALER_PATH = _ROOT / "models" / "gru" / "forecast_scaler.pkl"

WINDOW_SIZE: int = 50
CLIP_VAL: float = 5.0


def make_windows(
    df: pd.DataFrame,
    features: List[str],
    window: int = WINDOW_SIZE,
    step: int = 1,
    include_rul: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build sliding windows from a single engine's feature DataFrame.

    Args:
        df: Per-engine DataFrame sorted by cycle, with feature columns present.
        features: Ordered list of 56 feature column names.
        window: Number of cycles per window (default 50).
        step: Stride between windows (1 = dense, window = non-overlapping).
        include_rul: If True, return RUL targets for the last cycle of each window.

    Returns:
        X: float32 array of shape (N, window, len(features)).
        y: float32 array of shape (N,) if include_rul and 'rul' in df columns,
           else None.
    """
    df = df.sort_values("cycle").reset_index(drop=True)
    arr = df[features].values.astype(np.float32)
    T = len(arr)

    # Pad short engines to fill at least one window
    if T < window:
        pad = np.zeros((window - T, len(features)), dtype=np.float32)
        arr = np.vstack([pad, arr])
        T = window

    Xs, ys = [], []
    for start in range(0, T - window + 1, step):
        Xs.append(arr[start : start + window])
        if include_rul and "rul" in df.columns:
            # RUL at the last cycle of this window
            idx = min(start + window - 1, len(df) - 1)
            ys.append(float(df["rul"].iloc[idx]))

    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.float32) if ys else None
    return X, y


def scale_windows(X: np.ndarray) -> np.ndarray:
    """Apply the saved RobustScaler and clip to ±CLIP_VAL.

    Args:
        X: float32 array of shape (N, window, C).

    Returns:
        Scaled and clipped float32 array of the same shape.
    """
    scaler = joblib.load(_SCALER_PATH)
    N, T, C = X.shape
    X_flat = X.reshape(-1, C)
    X_scaled = scaler.transform(X_flat).reshape(N, T, C).astype(np.float32)
    return np.clip(X_scaled, -CLIP_VAL, CLIP_VAL)


def make_inference_window(
    df: pd.DataFrame,
    features: List[str],
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    """Build and scale a single inference window from an engine's recent history.

    Takes the last ``window`` cycles (or pads if shorter) and returns a
    scaled array of shape (1, window, len(features)) ready for the model.
    """
    df = df.sort_values("cycle")
    vals = df[features].values.astype(np.float32)
    T = len(vals)
    if T >= window:
        arr = vals[-window:]
    else:
        pad = np.zeros((window - T, len(features)), dtype=np.float32)
        arr = np.vstack([pad, vals])
    X = arr[np.newaxis, :, :]  # (1, window, n_features)
    return scale_windows(X)
