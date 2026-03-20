"""Engine Health MCP Server — 6 tools callable by the LLM agent.

Tools:
1. estimate_rul           — ForecastNet RUL prediction with confidence interval
2. detect_anomaly         — Sensor z-score check vs healthy-engine baseline
3. compute_degradation_rate — Fleet-relative degradation speed
4. get_stress_profile     — Operating condition history (how many high-stress flights?)
5. recommend_route        — What-if: RUL impact of a proposed route
6. schedule_maintenance   — Maintenance urgency given current RUL

Data source: data/processed/FD004_test.parquet for test engines.
             data/processed/FD004_train.parquet for fleet reference.

Run with: python -m src.mcp_server.server
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")

from fastmcp import FastMCP

_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once on first tool call
# ---------------------------------------------------------------------------
_reg = None
_det = None
_deg = None
_test_feat = None
_train_feat = None
_true_rul = None


def _get_reg():
    global _reg
    if _reg is None:
        from src.models.rul_regressor import RULRegressor
        _reg = RULRegressor.load()
    return _reg


def _get_det():
    global _det
    if _det is None:
        from src.models.anomaly_detector import AnomalyDetector
        _det = AnomalyDetector()
    return _det


def _get_deg():
    global _deg
    if _deg is None:
        from src.models.degradation_rate import DegradationRate
        _deg = DegradationRate.from_parquet()
    return _deg


def _get_test_feat():
    global _test_feat
    if _test_feat is None:
        from src.features.loader import load_processed
        from src.features.builder import build_features
        df = load_processed("FD004", "test")
        _test_feat = build_features(df, train_ref=_get_train_feat(), add_rolling=False)
    return _test_feat


def _get_train_feat():
    global _train_feat
    if _train_feat is None:
        from src.features.loader import load_processed
        from src.features.builder import build_features
        df = load_processed("FD004", "train")
        _train_feat = build_features(df, add_rolling=False)
    return _train_feat


def _get_true_rul():
    """Load ground-truth RUL for FD004 test engines (cached).

    Returns a dict mapping engine_id (1-indexed) to true remaining useful life
    from data/raw/RUL_FD004.txt.  Avoids the RUL_CAP=130 clip that affects
    model predictions.
    """
    global _true_rul
    if _true_rul is None:
        from src.features.loader import load_rul_labels
        series = load_rul_labels("FD004")
        _true_rul = {int(i + 1): int(v) for i, v in enumerate(series)}
    return _true_rul


def _get_engine(engine_id: int):
    """Return history for engine_id from test set (or train if not found)."""
    test = _get_test_feat()
    eng = test[test.unit == engine_id]
    if len(eng) == 0:
        train = _get_train_feat()
        eng = train[train.unit == engine_id]
    if len(eng) == 0:
        raise ValueError(f"Engine {engine_id} not found in FD004 test or train set.")
    return eng.sort_values("cycle")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("engine-health-tools")


@mcp.tool()
def estimate_rul(engine_id: int) -> Dict:
    """Estimate remaining useful life (RUL) for an engine using the ForecastNet ensemble.

    Args:
        engine_id: Engine unit number (1–248 in test set, 1–249 in train set).

    Returns:
        rul: Estimated cycles remaining (0–130 scale).
        lower_80: 10th-percentile of ensemble predictions (pessimistic bound).
        upper_80: 90th-percentile of ensemble predictions (optimistic bound).
        std: Cross-seed standard deviation (uncertainty indicator).
        current_cycle: Most recent observed cycle for this engine.
        accuracy: Model accuracy metrics for this RUL range (rmse, mae, bias).
        rul_true: Ground-truth RUL if available (test set only), else null.
    """
    from src.features.windows import make_inference_window
    from src.models.what_if import get_model_accuracy
    reg = _get_reg()
    eng = _get_engine(engine_id)
    X = make_inference_window(eng, reg.features)
    # Ensure we have a plain mutable dict for tool output (type checkers expect mapping
    # values to be assignable). Wrap the regressor output in dict() to avoid Pylance
    # complaining about specialized mapping types.
    result: Dict[str, Any] = dict(reg.predict(X))
    result["current_cycle"] = int(eng["cycle"].max())
    # Model accuracy for this RUL range
    result["accuracy"] = get_model_accuracy(result["rul"])
    # Ground truth if available
    last_rul = eng[eng["rul"].notna()]["rul"]
    rul_true_val: Optional[float] = float(last_rul.iloc[0]) if len(last_rul) > 0 else None
    result["rul_true"] = rul_true_val
    return result


@mcp.tool()
def detect_anomaly(engine_id: int, last_n_cycles: int = 20) -> Dict:
    """Check whether recent sensor readings deviate anomalously from the healthy fleet baseline.

    Useful for detecting broken or degraded sensors that may be producing misleading readings.
    A z-score above 2.5 on any sensor triggers a flag.

    Args:
        engine_id: Engine unit number.
        last_n_cycles: Number of recent cycles to analyse (default 20).

    Returns:
        anomalous: True if any sensor exceeds the threshold.
        z_scores: Per-sensor mean absolute deviation from healthy baseline.
        flagged_sensors: List of sensor names with z-score > 2.5.
        condition_cluster: Dominant operating condition in the analysis window.
        current_cycle: Most recent observed cycle.
    """
    det = _get_det()
    eng = _get_engine(engine_id)
    result = det.detect_latest(eng, last_n=last_n_cycles)
    result["current_cycle"] = int(eng["cycle"].max())
    return result


@mcp.tool()
def compute_degradation_rate(engine_id: int) -> Dict:
    """Compute how fast this engine is ageing compared to the fleet average.

    Compares the rolling slope of key degradation-indicator sensors over the
    last 20 cycles to the fleet median slope at the same lifecycle stage.

    Args:
        engine_id: Engine unit number.

    Returns:
        rate_multiplier: Fleet-relative rate (1.0 = average, 1.3 = 30% faster).
        trend_description: Human-readable summary.
        sensor_slopes: Per-sensor slopes over last 20 cycles.
        fleet_slopes: Fleet median slopes at this lifecycle stage.
        lifecycle_bin: Lifecycle fraction bin (0–9, where 9 = near end of life).
    """
    deg = _get_deg()
    eng = _get_engine(engine_id)
    return deg.compute(eng)


@mcp.tool()
def get_stress_profile(engine_id: int) -> Dict:
    """Analyse the operating condition history — how many high-stress flights has this engine flown?

    High-stress is defined as condition cluster 3, 4, or 5 (higher altitude/Mach).
    Low-stress is cluster 0, 1, or 2.

    Args:
        engine_id: Engine unit number.

    Returns:
        pct_high_stress: Fraction of cycles in high-stress conditions (0–1).
        condition_breakdown: Per-cluster cycle counts.
        total_cycles: Total observed cycles.
        dominant_condition: Most frequent condition cluster.
        stress_category: 'low' | 'moderate' | 'high'
    """
    eng = _get_engine(engine_id)
    if "condition" not in eng.columns:
        return {"error": "condition column not available"}

    total = len(eng)
    counts_raw = eng["condition"].value_counts().to_dict()
    counts = {int(k): int(v) for k, v in counts_raw.items()}  # type: ignore[arg-type]

    high_stress_cycles = sum(v for k, v in counts.items() if k >= 3)
    pct_high = high_stress_cycles / total if total > 0 else 0.0
    dominant = int(eng["condition"].mode().iloc[0])  # type: ignore[arg-type]

    if pct_high < 0.3:
        stress_cat = "low"
    elif pct_high < 0.6:
        stress_cat = "moderate"
    else:
        stress_cat = "high"

    return {
        "pct_high_stress": round(pct_high, 3),
        "condition_breakdown": counts,
        "total_cycles": total,
        "dominant_condition": dominant,
        "stress_category": stress_cat,
    }


@mcp.tool()
def recommend_route(engine_id: int, proposed_cycles: int, condition: int) -> Dict:
    """Project the RUL impact of assigning this engine to a route.

    Simulates the additional cycles at the proposed operating condition and
    re-runs the RUL model to estimate remaining life after the route.

    Args:
        engine_id: Engine unit number.
        proposed_cycles: Number of additional flight cycles in the proposed route.
        condition: Operating condition cluster for the route (0–5).
            0–2 = low stress (short-haul / low altitude).
            3–5 = high stress (long-haul / high altitude / high Mach).

    Returns:
        baseline_rul: Predicted RUL before the route.
        projected_rul: Predicted RUL after the proposed route.
        rul_change: Change in RUL (negative = life consumed faster than expected).
        recommendation: Plain-text operational recommendation.
        stress_factor: Relative stress of the proposed condition vs fleet average.
    """
    from src.models.what_if import project_rul
    reg = _get_reg()
    eng = _get_engine(engine_id)
    rate = _get_deg().compute(eng).get("rate_multiplier", 1.0)
    return project_rul(eng, n_cycles=proposed_cycles, condition=condition,
                       regressor=reg, rate_multiplier=rate)


@mcp.tool()
def schedule_maintenance(engine_id: int) -> Dict:
    """Determine maintenance urgency based on current RUL estimate.

    Computes RUL and returns a maintenance flag with urgency level and
    suggested action window.

    Args:
        engine_id: Engine unit number.

    Returns:
        flag: True if maintenance is recommended within 20 cycles.
        urgency: 'IMMEDIATE' | 'HIGH' | 'MODERATE' | 'LOW'
        rul: Current RUL estimate.
        suggested_within_cycles: Recommended action window in cycles.
        reasoning: Explanation of the urgency classification.
    """
    from src.features.windows import make_inference_window
    reg = _get_reg()
    eng = _get_engine(engine_id)
    X = make_inference_window(eng, reg.features)
    pred = reg.predict(X)
    rul = pred["rul"]
    lower = pred["lower_80"]

    if lower < 10 or rul < 15:
        urgency = "IMMEDIATE"
        within = 5
        flag = True
        reason = (
            f"RUL {rul:.0f} — ground immediately. High risk of engine failure. "
            "Full inspection and service required before any further operation."
        )
    elif rul < 30:
        urgency = "HIGH"
        within = 10
        flag = True
        reason = (
            f"RUL {rul:.0f}. Approaching end of serviceable life — "
            "schedule maintenance now to remain above safety threshold (6 RUL)."
        )
    elif rul < 60:
        urgency = "MODERATE"
        within = 20
        flag = True
        reason = (
            f"RUL {rul:.0f}. Service recommended to maintain "
            "operational readiness and avoid urgent grounding."
        )
    else:
        urgency = "LOW"
        within = max(20, int(rul * 0.5))
        flag = False
        reason = f"RUL {rul:.0f}. No immediate maintenance required."

    return {
        "flag": flag,
        "urgency": urgency,
        "rul": round(rul, 1),
        "lower_80": round(lower, 1),
        "suggested_within_cycles": within,
        "reasoning": reason,
    }


@mcp.tool()
def find_replacement_engine(
    min_rul: float = 60,
    max_results: int = 5,
    exclude_engines: str = "",
    target_engine_id: int = 0,
) -> Dict:
    """Search the test fleet for condition-compatible engines with high RUL.

    When target_engine_id is provided, filters candidates to engines that
    operate in a compatible stress category (low-stress 0-2 vs high-stress 3-5)
    and prioritises engines with the same dominant operating condition.

    Args:
        min_rul: Minimum true RUL to consider (default 60).
        max_results: Maximum number of candidates to return (default 5).
        exclude_engines: Comma-separated engine IDs to exclude (e.g. '12,42').
        target_engine_id: If > 0, automatically exclude this engine and filter
            candidates by compatible operating conditions. Default 0.

    Returns:
        candidates: List of {engine_id, predicted_rul, current_cycle,
            dominant_condition, dominant_condition_label, condition_match}
            sorted by condition match then RUL descending.
        total_searched: Number of engines evaluated.
        total_qualifying: Number meeting all criteria.
    """
    from src.models.what_if import CONDITION_LABELS

    true_ruls = _get_true_rul()
    test = _get_test_feat()
    units = sorted(test["unit"].unique())

    exclude = set()
    if exclude_engines:
        exclude = {int(x.strip()) for x in exclude_engines.split(",") if x.strip()}
    if target_engine_id > 0:
        exclude.add(target_engine_id)

    # Get target engine's dominant condition for matching
    target_dominant = None
    target_stress_high = None
    if target_engine_id > 0:
        target_eng = test[test["unit"] == target_engine_id]
        if "condition" in target_eng.columns and len(target_eng) > 0:
            target_dominant = int(target_eng["condition"].mode().iloc[0])
            target_stress_high = target_dominant >= 3

    candidates = []
    for uid in units:
        uid_int = int(uid)
        if uid_int in exclude:
            continue
        rul = true_ruls.get(uid_int)
        if rul is None or rul < min_rul:
            continue

        eng = test[test["unit"] == uid]
        cand_dominant = None
        cand_label = None
        condition_match = True  # default if no target specified

        if "condition" in eng.columns and len(eng) > 0:
            cand_dominant = int(eng["condition"].mode().iloc[0])
            cand_label = CONDITION_LABELS.get(cand_dominant, f"Condition {cand_dominant}")

            # Filter: must be same stress category (low vs high)
            if target_stress_high is not None:
                cand_stress_high = cand_dominant >= 3
                if cand_stress_high != target_stress_high:
                    continue
                condition_match = (cand_dominant == target_dominant)

        candidates.append({
            "engine_id": uid_int,
            "predicted_rul": rul,
            "current_cycle": int(eng["cycle"].max()),
            "dominant_condition": cand_dominant,
            "dominant_condition_label": cand_label,
            "condition_match": condition_match,
        })

    # Sort: exact condition match first, then by RUL
    candidates.sort(
        key=lambda c: (not c.get("condition_match", False), -c["predicted_rul"])
    )

    return {
        "candidates": candidates[:max_results],
        "total_searched": len(units) - len(exclude),
        "total_qualifying": len(candidates),
    }


@mcp.tool()
def diagnose_engine(engine_id: int) -> Dict:
    """Root-cause analysis: explains WHY an engine is degrading.

    Combines anomaly detection, degradation rate, stress profile, and
    per-sensor trend analysis to determine whether degradation is caused by:
    - High-stress operating conditions (too many harsh flights)
    - Sensor malfunction (broken/miscalibrated sensor giving false readings)
    - Accelerated component wear (specific parts wearing faster than fleet)
    - Normal operational wear

    Use this when you need to explain degradation, not just measure it.

    Args:
        engine_id: Engine unit number.

    Returns:
        primary_cause: Main driver of degradation.
        all_causes: All identified causes.
        contributing_factors: Detailed explanations.
        sensor_findings: Per-sensor analysis with category and explanation.
        risk_level: CRITICAL / HIGH / MODERATE / LOW.
        summary: Plain-English diagnosis.
        recommended_actions: List of actionable steps.
    """
    from src.models.diagnosis import diagnose

    eng = _get_engine(engine_id)

    # Gather all evidence
    rul_result = estimate_rul(engine_id)
    anomaly_result = detect_anomaly(engine_id, last_n_cycles=20)
    degradation_result = compute_degradation_rate(engine_id)
    stress_result = get_stress_profile(engine_id)

    return diagnose(eng, anomaly_result, degradation_result, stress_result, rul_result)


@mcp.tool()
def compare_conditions(engine_id: int, cycles: int = 20) -> Dict:
    """Compare projected RUL outcomes under ALL operating conditions (0–5).

    Projects what would happen if this engine flies N additional cycles at
    each condition level. Answers: "Which route type gives us the best outcome?"

    Args:
        engine_id: Engine unit number.
        cycles: Number of proposed flight cycles to simulate (default 20).

    Returns:
        baseline_rul: Current predicted RUL before any route.
        projections: List of {condition, projected_rul, rul_change, stress_factor,
            recommendation} sorted from best to worst outcome.
        best_condition: The condition that preserves the most RUL.
        worst_condition: The condition that consumes the most RUL.
        spread: Difference in RUL between best and worst condition.
    """
    from src.models.what_if import project_rul, CONDITION_LABELS, CONDITION_SEVERITY
    from src.models.what_if import RUL_SOFT_FLOOR, RUL_HARD_FLOOR

    reg = _get_reg()
    eng = _get_engine(engine_id)
    rate = _get_deg().compute(eng).get("rate_multiplier", 1.0)

    projections = []
    baseline: float = 0.0
    for cond in range(6):
        result = project_rul(eng, n_cycles=cycles, condition=cond,
                             regressor=reg, rate_multiplier=rate)
        if cond == 0:
            baseline = result["baseline_rul"]
        projections.append({
            "condition": cond,
            "label": CONDITION_LABELS.get(cond, f"Condition {cond}"),
            "severity": CONDITION_SEVERITY.get(cond, "unknown"),
            "projected_rul": result["projected_rul"],
            "rul_change": result["rul_change"],
            "recommendation": result["recommendation"],
        })

    # Compute per-condition flight feasibility
    usable_rul = max(baseline - RUL_SOFT_FLOOR, 0)
    for p in projections:
        rul_cost = abs(p["rul_change"]) if p["rul_change"] != 0 else 0.01
        rul_per_cycle = rul_cost / cycles
        flights_remaining = int(usable_rul / rul_per_cycle) if rul_per_cycle > 0 else 999
        p["rul_cost_per_cycle"] = round(rul_per_cycle, 2)
        p["flights_before_rul_10"] = flights_remaining
        p["safe_for_journey"] = p["projected_rul"] > RUL_SOFT_FLOOR
        # Three-tier status
        if p["projected_rul"] > RUL_SOFT_FLOOR:
            p["status"] = "safe"
        elif p["projected_rul"] > RUL_HARD_FLOOR:
            p["status"] = "at-risk"
        else:
            p["status"] = "grounded"

    projections.sort(key=lambda p: p["projected_rul"], reverse=True)

    best = projections[0]
    worst = projections[-1]
    spread = round(best["projected_rul"] - worst["projected_rul"], 1)
    best_label = CONDITION_LABELS.get(best["condition"], str(best["condition"]))
    worst_label = CONDITION_LABELS.get(worst["condition"], str(worst["condition"]))

    return {
        "baseline_rul": round(baseline, 1),
        "rul_floor": RUL_SOFT_FLOOR,
        "rul_hard_floor": RUL_HARD_FLOOR,
        "usable_rul": round(usable_rul, 1),
        "cycles_simulated": cycles,
        "projections": projections,
        "best_condition": best["condition"],
        "worst_condition": worst["condition"],
        "spread": spread,
        "summary": (
            f"Current RUL: {baseline:.0f} cycles (usable above threshold "
            f"{RUL_SOFT_FLOOR}: {usable_rul:.0f} cycles). "
            f"Best: {best_label} costs ~{best['rul_cost_per_cycle']:.2f} RUL/cycle "
            f"({best['flights_before_rul_10']} flights left). "
            f"Worst: {worst_label} costs ~{worst['rul_cost_per_cycle']:.2f} RUL/cycle "
            f"({worst['flights_before_rul_10']} flights left)."
        ),
    }


@mcp.tool()
def find_best_engine_for_mission(
    mission_cycles: int,
    mission_condition: int = 3,
    min_post_mission_rul: float = 30,
    max_candidates: int = 5,
) -> Dict:
    """Find the best engine for a specific mission, considering post-mission RUL.

    Unlike find_replacement_engine (which just ranks by current RUL), this tool
    simulates the actual mission on each candidate and ranks by predicted RUL
    AFTER the mission. An engine with high current RUL but fast degradation may
    rank lower than one with moderate RUL but resilience to the mission conditions.

    Args:
        mission_cycles: Number of flight cycles in the mission.
        mission_condition: Operating condition for the mission (0–5, default 3).
        min_post_mission_rul: Minimum acceptable RUL after completion (default 30).
        max_candidates: Max candidates to return (default 5).

    Returns:
        candidates: List of {engine_id, current_rul, post_mission_rul, rul_consumed,
            degradation_rate, stress_history, suitability} sorted by post_mission_rul.
        mission: Summary of mission parameters.
        total_evaluated: Engines tested.
        total_suitable: Engines meeting the min_post_mission_rul threshold.
    """
    from src.models.what_if import project_rul
    from src.features.windows import make_inference_window

    reg = _get_reg()
    test = _get_test_feat()
    units = sorted(test["unit"].unique())

    candidates = []
    for uid in units:
        try:
            eng = test[test.unit == uid].sort_values("cycle")

            # Current RUL
            X = make_inference_window(eng, reg.features)
            pred = reg.predict(X)
            current_rul = pred["rul"]

            # Skip engines that are already too low
            if current_rul < mission_cycles + min_post_mission_rul * 0.5:
                continue

            # Get additional context
            deg = _get_deg().compute(eng)
            rate = deg.get("rate_multiplier", 1.0)
            stress = get_stress_profile(int(uid))

            # Project post-mission RUL
            proj = project_rul(eng, n_cycles=mission_cycles,
                               condition=mission_condition, regressor=reg,
                               rate_multiplier=rate)
            post_rul = proj["projected_rul"]

            rul_consumed = round(current_rul - post_rul, 1)
            efficiency = round(rul_consumed / mission_cycles, 2) if mission_cycles > 0 else 0

            if post_rul > 60 and deg["rate_multiplier"] < 1.2:
                suitability = "excellent"
            elif post_rul > 40 and deg["rate_multiplier"] < 1.4:
                suitability = "good"
            elif post_rul >= min_post_mission_rul:
                suitability = "acceptable"
            else:
                suitability = "marginal"

            candidates.append({
                "engine_id": int(uid),
                "current_rul": round(current_rul, 1),
                "post_mission_rul": round(post_rul, 1),
                "rul_consumed": rul_consumed,
                "consumption_ratio": efficiency,
                "degradation_rate": deg["rate_multiplier"],
                "stress_history": stress["stress_category"],
                "suitability": suitability,
            })
        except Exception:
            continue

    candidates.sort(key=lambda c: c["post_mission_rul"], reverse=True)

    return {
        "candidates": candidates[:max_candidates],
        "mission": {
            "cycles": mission_cycles,
            "condition": mission_condition,
            "condition_label": "low-stress" if mission_condition <= 2 else "high-stress",
            "min_post_mission_rul": min_post_mission_rul,
        },
        "total_evaluated": len(units),
        "total_suitable": len(candidates),
    }


@mcp.tool()
def forecast_journey(engine_id: int, legs: str) -> Dict:
    """Forecast RUL through a multi-leg journey sequence.

    Simulates a sequence of flight legs at different operating conditions
    and shows cumulative RUL impact after each leg.  Use this to answer
    questions like "Can this engine handle condition 5 outbound then
    condition 0 return?"

    Args:
        engine_id: Engine unit number.
        legs: JSON array of flight legs. Each element needs "condition" (0-5)
            and "cycles" (number of flights). Example:
            '[{"condition": 5, "cycles": 10}, {"condition": 0, "cycles": 10}]'

    Returns:
        baseline_rul: RUL before the journey.
        legs: Per-leg results with label, severity, rul_after.
        final_rul: Predicted RUL after the complete journey.
        feasible: True if final RUL > hard safety cutoff (6).
        safe: True if final RUL > maintenance threshold (10).
        recommendation: Plain-text assessment.
    """
    import json as _json
    from src.models.what_if import project_journey

    reg = _get_reg()
    eng = _get_engine(engine_id)
    rate = _get_deg().compute(eng).get("rate_multiplier", 1.0)

    try:
        parsed_legs = _json.loads(legs)
    except (ValueError, TypeError) as exc:
        return {"error": f"Could not parse legs JSON: {exc}"}

    return project_journey(eng, parsed_legs, regressor=reg,
                           rate_multiplier=rate)


def fleet_health_summary(flag_degradation: bool = False) -> dict:
    """Bucket all 248 test engines by true RUL health tier.

    Uses ground-truth RUL from RUL_FD004.txt (no model inference, no RUL_CAP=130 clip).
    Runs in ~0.2s for the full fleet.

    Tiers:
      GROUNDED : RUL ≤ 6   (hard safety floor — no flights permitted)
      CRITICAL : 6 < RUL ≤ 10
      ADVISORY : 10 < RUL ≤ 30
      HEALTHY  : RUL > 30

    Args:
        flag_degradation: If True, compute degradation rate for GROUNDED and
            CRITICAL engines and flag those with rate_multiplier > 1.3.

    Returns:
        buckets: dict with keys 'GROUNDED', 'CRITICAL', 'ADVISORY', 'HEALTHY',
            each a list of {engine_id, rul, [rate_multiplier]} sorted by RUL asc.
        counts: per-tier engine counts.
        total: total engines evaluated.
    """
    true_ruls = _get_true_rul()

    buckets: dict = {"GROUNDED": [], "CRITICAL": [], "ADVISORY": [], "HEALTHY": []}

    for eid, rul in true_ruls.items():
        if rul <= 6:
            tier = "GROUNDED"
        elif rul <= 10:
            tier = "CRITICAL"
        elif rul <= 30:
            tier = "ADVISORY"
        else:
            tier = "HEALTHY"
        buckets[tier].append({"engine_id": eid, "rul": rul})

    # Sort each tier by RUL ascending (most urgent first)
    for tier in buckets:
        buckets[tier].sort(key=lambda e: e["rul"])

    # Optionally flag abnormal degradation for at-risk engines
    if flag_degradation:
        for tier in ("GROUNDED", "CRITICAL"):
            for entry in buckets[tier]:
                try:
                    eng = _get_engine(entry["engine_id"])
                    rate = _get_deg().compute(eng).get("rate_multiplier", 1.0)
                    entry["rate_multiplier"] = round(rate, 2)
                except Exception:
                    pass

    counts = {tier: len(entries) for tier, entries in buckets.items()}
    return {"buckets": buckets, "counts": counts, "total": len(true_ruls)}


if __name__ == "__main__":
    mcp.run()
