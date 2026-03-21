"""Centralised configuration constants for Engine Health Agent.

All tunable thresholds and shared numeric constants live here.
Import from this module — do not duplicate values in individual modules.

Note: pkl-derived values (e.g. fleet_median_life, fleet_condition_slopes) are
intentionally NOT loaded here to keep this module side-effect-free and
importable anywhere in the stack without triggering file I/O.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Shared rolling window size (cycles) — used by builder.py AND degradation_rate.py.
# Both apply this window to the same sensor data; they must always be equal.
ROLL_WINDOW: int = 20

# ---------------------------------------------------------------------------
# Drift detection (Population Stability Index thresholds)
# ---------------------------------------------------------------------------

PSI_MODERATE: float = 0.10     # worth monitoring
PSI_SIGNIFICANT: float = 0.25  # model accuracy may be degraded — investigate

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

Z_THRESHOLD: float = 2.5  # mean absolute z-score above which a sensor is flagged

# ---------------------------------------------------------------------------
# RUL safety floors (cycles remaining)
# ---------------------------------------------------------------------------

RUL_HARD_FLOOR: int = 6   # ground the aircraft immediately
RUL_SOFT_FLOOR: int = 10  # schedule maintenance before next flight

# ---------------------------------------------------------------------------
# Dataset bounds (NASA CMAPSS FD004 test set)
# ---------------------------------------------------------------------------

TEST_ENGINE_MAX: int = 248   # highest valid test-set engine unit number
CONDITION_MAX: int = 5       # highest valid operating condition index (0–5)
