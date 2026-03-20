"""Feature builder — produces the 56-feature set used by ForecastNet v5c.

Feature families:
- n_s*          : condition-normalised sensor readings (primary signal)
- roll_mean_s*  : 20-cycle rolling mean per engine
- roll_std_s*   : 20-cycle rolling std per engine
- slope_s*      : OLS slope over last 20 cycles per engine
- exp_std_s*    : expanding std from engine start (accumulated volatility)
- ewma_s*       : EWMA (span=20) for s3, s4, s11, s14
- cycle_norm    : cycle / fleet_median_life (non-leaky life-stage proxy)
- cx_s*         : cycle_norm × roll_mean for s3, s11, s14
- cfd_s*        : fleet lifecycle deviation for s3, s4, s11, s13
- os1/os2/os3   : raw operating settings (contextual)

Four features excluded via permutation importance:
    slope_s3, n_s3, roll_std_s15, roll_std_s4
"""

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from src.features.normaliser import SENSORS

_ROOT = Path(__file__).resolve().parents[2]
_PREP = _ROOT / "models" / "preprocessing"

FLEET_MEDIAN_LIFE: int = 239  # used by ForecastNet (matches checkpoint config)
ROLL_WINDOW: int = 20
_EWMA_SENSORS = ["s11", "s14", "s3", "s4"]
_CX_SENSORS = ["s11", "s14", "s3"]
_CFD_SENSORS = ["s4", "s13", "s11", "s3"]
_N_CFD_BINS = 20

DROP_FEATURES = {"slope_s3", "n_s3", "roll_std_s15", "roll_std_s4"}


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    arr = x.values
    slopes = np.empty(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        seg = arr[start : i + 1]
        if len(seg) < 3:
            slopes[i] = 0.0
        else:
            t = np.arange(len(seg), dtype=float)
            slopes[i] = np.polyfit(t, np.asarray(seg, dtype=np.float64), 1)[0]
    return pd.Series(slopes, index=x.index)


def compute_rolling_features(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Add roll_mean_s*, roll_std_s*, slope_s* columns per engine."""
    df = df.copy().sort_values(["unit", "cycle"])
    for s in SENSORS:
        col = f"n_{s}"
        if col not in df.columns:
            continue
        grp = df.groupby("unit")[col]
        df[f"roll_mean_{s}"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{s}"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"slope_{s}"] = grp.transform(
            lambda x: _rolling_slope(x, window)
        )
    return df


def compute_exp_std(df: pd.DataFrame) -> pd.DataFrame:
    """Add exp_std_s* — expanding std from engine start per engine."""
    df = df.copy()
    for s in SENSORS:
        col = f"n_{s}"
        if col not in df.columns:
            continue
        df[f"exp_std_{s}"] = df.groupby("unit")[col].transform(
            lambda x: x.expanding(min_periods=2).std().fillna(0)
        )
    return df


def add_derived_features(
    df: pd.DataFrame,
    train_ref: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add EWMA, cycle_norm, cx_*, cfd_*, and os1/os2/os3 alias columns.

    Args:
        df: DataFrame with n_s* and roll_mean_s* columns.
        train_ref: Training DataFrame used to compute fleet lifecycle means for
            CFD.  If None, uses df itself (only safe for the training split).
    """
    df = df.copy()
    ref = train_ref if train_ref is not None else df

    # EWMA (span=20) for key sensors
    for s in _EWMA_SENSORS:
        col = f"n_{s}"
        if col in df.columns:
            df[f"ewma_{s}"] = df.groupby("unit")[col].transform(
                lambda x: x.ewm(span=20, min_periods=1).mean()
            )

    # Life-stage proxy (non-leaky)
    df["cycle_norm"] = df["cycle"] / FLEET_MEDIAN_LIFE

    # Cycle × rolling-mean interactions
    for s in _CX_SENSORS:
        col = f"roll_mean_{s}"
        if col in df.columns:
            df[f"cx_{s}"] = df["cycle_norm"] * df[col]

    # Cycle-bin fleet deviation (CFD)
    max_cyc = int(ref["cycle"].max())
    edges = list(range(0, max_cyc + _N_CFD_BINS + 1, _N_CFD_BINS))
    ref_copy = ref.copy()
    ref_copy["_cb"] = pd.cut(ref_copy["cycle"], bins=edges, labels=False)
    df["_cb"] = pd.cut(df["cycle"], bins=edges, labels=False)
    for s in _CFD_SENSORS:
        col = f"n_{s}"
        if col in df.columns:
            fleet_mean = ref_copy.groupby("_cb")[col].mean()
            df[f"cfd_{s}"] = df[col] - df["_cb"].map(fleet_mean).fillna(0)
    df = df.drop(columns=["_cb"])

    # Operating settings as contextual features (os1/os2/os3 naming matches checkpoint)
    for i, src_col in enumerate(["op1", "op2", "op3"], start=1):
        if src_col in df.columns and f"os{i}" not in df.columns:
            df[f"os{i}"] = df[src_col]

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the 56 features used by ForecastNet v5c, sorted alphabetically."""
    exclude = {"unit", "cycle", "rul", "cycle_frac", "condition", "dataset"}
    exclude.update({"op1", "op2", "op3", "s1", "s2", "s3", "s4", "s5", "s6",
                    "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
                    "s15", "s16", "s17", "s18", "s19", "s20", "s21"})
    return sorted(
        c for c in df.columns
        if c not in exclude
        and c not in DROP_FEATURES
        and float(df[c].std()) > 1e-12
    )


def build_features(
    df: pd.DataFrame,
    train_ref: Optional[pd.DataFrame] = None,
    add_rolling: bool = True,
) -> pd.DataFrame:
    """Full feature pipeline: raw/parquet → 56-feature DataFrame.

    Args:
        df: Input DataFrame.  Must have at minimum [unit, cycle, n_s*, op1/op2/op3].
            If ``add_rolling=False``, must already contain roll_mean_s*, roll_std_s*,
            slope_s* (i.e. loaded from the processed parquet).
        train_ref: Training DataFrame for CFD fleet baseline.  Pass when building
            features for test/validation data.
        add_rolling: If True, compute rolling mean/std/slope from n_s* columns.
            Set to False when df already has these from the processed parquet.

    Returns:
        DataFrame with the 56 ForecastNet features (and unit/cycle/rul if present).
    """
    df = df.copy().sort_values(["unit", "cycle"])
    if add_rolling:
        df = compute_rolling_features(df)
    df = compute_exp_std(df)
    df = add_derived_features(df, train_ref=train_ref)
    return df
