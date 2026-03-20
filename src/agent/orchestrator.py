"""Engine Health Agent — LangChain ReAct agent with local Llama 3.1 8B.

Uses a text-based ReAct loop (Thought/Action/Observation) for reliable
tool calling with the local model.  No Ollama dependency.

Usage::

    from src.agent.orchestrator import AgentOrchestrator
    agent = AgentOrchestrator()
    agent.load()
    result = agent.run(engine_id=42)
    print(result["recommendation"])
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool as langchain_tool

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models" / "llama-3.1-8b-instruct"
_RUNS_FP = _ROOT / "outputs" / "agent_runs.jsonl"
_RUNS_FP.parent.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 8

# ---------------------------------------------------------------------------
# Model loading  (singleton)
# ---------------------------------------------------------------------------

_chat_singleton = None


def _resolve_model_path(base: Path) -> Path:
    refs = base / "refs" / "main"
    if refs.exists():
        rev = refs.read_text().strip()
        snap = base / "snapshots" / rev
        if snap.exists():
            return snap
    return base


def load_llm(print_fn: Callable[..., None] = print):
    """Load Llama 3.1 8B Instruct in 4-bit, return ChatHuggingFace (cached)."""
    global _chat_singleton
    if _chat_singleton is not None:
        return _chat_singleton

    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import pipeline as hf_pipeline

    model_path = str(_resolve_model_path(_MODEL_DIR))
    print_fn("  Loading Llama 3.1 8B Instruct (4-bit) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="cuda:0",
        max_memory={0: "14GiB"},
        low_cpu_mem_usage=True,
    )
    torch.cuda.empty_cache()

    # Configure generation — set on model config to avoid deprecation warnings
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = 1024

    # Suppress noisy transformers generation warnings
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
    warnings.filterwarnings("ignore", message=".*generation_config.*deprecated.*")
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(
        pipeline=pipe,
        pipeline_kwargs={"max_new_tokens": 1024},
    )
    _chat_singleton = ChatHuggingFace(llm=llm)
    print_fn("  Model loaded (4-bit, ~5 GB VRAM).")
    return _chat_singleton


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@langchain_tool
def estimate_rul(engine_id: int) -> str:
    """Estimate Remaining Useful Life (RUL) for an engine.
    Returns rul, lower_80, upper_80, std, current_cycle."""
    from src.mcp_server.server import estimate_rul as fn
    return json.dumps(fn(engine_id=engine_id), indent=2, default=str)


@langchain_tool
def compute_degradation_rate(engine_id: int) -> str:
    """Compute how fast this engine is ageing vs fleet average.
    rate_multiplier 1.0 = average. Above 1.3 = abnormal. Below 0.7 = slow."""
    from src.mcp_server.server import compute_degradation_rate as fn
    return json.dumps(fn(engine_id=engine_id), indent=2, default=str)


@langchain_tool
def detect_anomaly(engine_id: int, last_n_cycles: int = 20) -> str:
    """Check whether recent sensor readings are anomalous vs the healthy
    fleet baseline. Returns anomalous, z_scores, flagged_sensors."""
    from src.mcp_server.server import detect_anomaly as fn
    return json.dumps(
        fn(engine_id=engine_id, last_n_cycles=last_n_cycles),
        indent=2, default=str,
    )


@langchain_tool
def get_stress_profile(engine_id: int) -> str:
    """Analyse operating condition history.
    Returns pct_high_stress, stress_category, condition_breakdown."""
    from src.mcp_server.server import get_stress_profile as fn
    return json.dumps(fn(engine_id=engine_id), indent=2, default=str)


@langchain_tool
def diagnose_engine(engine_id: int) -> str:
    """Root-cause analysis: WHY is the engine degrading.
    Returns primary_cause, risk_level, sensor_findings, actions, summary."""
    from src.mcp_server.server import diagnose_engine as fn
    return json.dumps(fn(engine_id=engine_id), indent=2, default=str)


@langchain_tool
def compare_conditions(engine_id: int, cycles: int = 10) -> str:
    """Project RUL under all 6 operating conditions (0-5).
    Returns projections with flights_before_rul_10 for each condition,
    best/worst condition, and whether each condition is safe."""
    from src.mcp_server.server import compare_conditions as fn
    return json.dumps(
        fn(engine_id=engine_id, cycles=cycles), indent=2, default=str,
    )


@langchain_tool
def schedule_maintenance(engine_id: int) -> str:
    """Determine maintenance urgency.
    Returns urgency (IMMEDIATE/HIGH/MODERATE/LOW), suggested_within_cycles."""
    from src.mcp_server.server import schedule_maintenance as fn
    return json.dumps(fn(engine_id=engine_id), indent=2, default=str)


@langchain_tool
def recommend_route(engine_id: int, proposed_cycles: int, condition: int) -> str:
    """Project RUL impact of a specific route. condition 0-5.
    Returns baseline_rul, projected_rul, rul_change."""
    from src.mcp_server.server import recommend_route as fn
    return json.dumps(
        fn(engine_id=engine_id, proposed_cycles=proposed_cycles,
           condition=condition), indent=2, default=str,
    )


@langchain_tool
def find_replacement_engine(
    min_rul: float = 60.0,
    max_results: int = 5,
    target_engine_id: int = 0,
) -> str:
    """Search fleet for engines with highest RUL for replacement.
    Returns candidates [{engine_id, predicted_rul}].
    Pass target_engine_id to restrict to condition-compatible engines."""
    from src.mcp_server.server import find_replacement_engine as fn
    return json.dumps(
        fn(min_rul=min_rul, max_results=max_results,
           target_engine_id=target_engine_id),
        indent=2, default=str,
    )


@langchain_tool
def find_best_engine_for_mission(
    mission_cycles: int, mission_condition: int = 3,
    min_post_mission_rul: float = 30.0, max_candidates: int = 5,
) -> str:
    """Find best engine for a mission. Ranks by post-mission RUL.
    Returns ranked candidates with suitability rating."""
    from src.mcp_server.server import find_best_engine_for_mission as fn
    return json.dumps(
        fn(mission_cycles=mission_cycles, mission_condition=mission_condition,
           min_post_mission_rul=min_post_mission_rul,
           max_candidates=max_candidates), indent=2, default=str,
    )


@langchain_tool
def forecast_journey(engine_id: int, legs: str) -> str:
    """Forecast RUL through a multi-leg journey sequence.
    legs: JSON array e.g. '[{"condition": 5, "cycles": 10}, {"condition": 0, "cycles": 10}]'
    Returns per-leg RUL and whether the journey is feasible."""
    from src.mcp_server.server import forecast_journey as fn
    return json.dumps(fn(engine_id=engine_id, legs=legs), indent=2, default=str)


_TOOL_MAP: Dict[str, Any] = {}


def _get_tool_map() -> Dict[str, Any]:
    if not _TOOL_MAP:
        for t in [
            estimate_rul, compute_degradation_rate, detect_anomaly,
            get_stress_profile, diagnose_engine, compare_conditions,
            schedule_maintenance, recommend_route,
            find_replacement_engine, find_best_engine_for_mission,
            forecast_journey,
        ]:
            _TOOL_MAP[t.name] = t
    return _TOOL_MAP


# ---------------------------------------------------------------------------
# System prompt with ReAct format
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    tool_descs = []
    for t in _get_tool_map().values():
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        props = schema.get("properties", {})
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in props.items()
        )
        tool_descs.append(f"- {t.name}({params}): {t.description}")
    tool_text = "\n".join(tool_descs)

    return f"""\
You are an aircraft engine health analyst for NASA CMAPSS turbofan engines.
You have tools that query a predictive maintenance system.

CONTEXT:
- Each "cycle" = one flight (takeoff + cruise + landing).
- RUL = remaining useful life in cycles. RUL below 10 means the engine must be grounded.
- Operating conditions 0-2 are low-stress. Conditions 3-5 are high-stress.

PROTOCOL — call these tools IN ORDER:
1. estimate_rul — get current RUL and confidence interval.
2. compute_degradation_rate — check if degradation is normal (0.7-1.3x) or abnormal.
3. compare_conditions — see how many flights remain per condition before hitting RUL 10. Use cycles=10.
4. IF degradation is abnormal: call diagnose_engine for root cause.
5. IF RUL < 30: call schedule_maintenance.
6. IF engine must be grounded (all conditions unsafe): call find_replacement_engine.

After calling at least 3 tools, give your Final Answer as a brief recommendation:
- State whether the engine is safe to fly and under which conditions.
- If grounding is needed, say so and suggest a replacement.
- Mention maintenance timeline if applicable.
- Do NOT repeat raw numbers — the structured data section is built automatically from tool outputs.

AVAILABLE TOOLS:
{tool_text}

FORMAT:
Thought: <reasoning>
Action: <tool_name>
Action Input: {{"param": value}}

After Observation, continue with Thought/Action or:
Final Answer: <your recommendation>

You MUST call at least 3 tools before Final Answer. Begin."""


# ---------------------------------------------------------------------------
# ReAct parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"Action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_ACTION_INPUT_RE = re.compile(
    r"Action\s*Input\s*:\s*(.+)",
    re.IGNORECASE,
)
_FINAL_RE = re.compile(
    r"Final\s*Answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL
)


def _parse_action(text: str) -> Optional[tuple]:
    m = _ACTION_RE.search(text)
    if not m:
        return None
    tool_name = m.group(1).strip().lower().replace(" ", "_")

    im = _ACTION_INPUT_RE.search(text)
    if not im:
        return None
    raw = im.group(1).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return tool_name, raw


def _parse_final(text: str) -> Optional[str]:
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def _exec_tool(name: str, raw_input: str, print_fn: Callable) -> str:
    """Parse input and execute a tool."""
    tools = _get_tool_map()

    # Exact match first, then fuzzy match
    if name not in tools:
        # Try to find a match (handles hyphens, dots, extra prefixes)
        cleaned = re.sub(r"[^a-z0-9_]", "_", name)
        match = None
        for tname in tools:
            if tname == cleaned or tname in name or name in tname:
                match = tname
                break
        if match:
            name = match
        else:
            return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(tools)}"})

    # Parse raw_input to a dict
    raw = raw_input.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()

    try:
        params = json.loads(raw)
        if isinstance(params, (int, float)):
            params = {"engine_id": int(params)}
    except (json.JSONDecodeError, ValueError):
        try:
            params = {"engine_id": int(raw)}
        except ValueError:
            return json.dumps({"error": f"Could not parse: {raw_input}"})

    for k, v in list(params.items()):
        if isinstance(v, float) and v == int(v):
            params[k] = int(v)

    print_fn(f"\n  -> {name}({json.dumps(params)})")

    try:
        result = tools[name].invoke(params)
        _print_tool_summary(result, print_fn)
        return result
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _print_tool_summary(output: str, print_fn: Callable) -> None:
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return
    if "rul" in data and "rate_multiplier" not in data and "risk_level" not in data and "flag" not in data:
        acc = data.get("accuracy", {})
        mae_str = f"  (±{acc['mae']:.0f} MAE)" if acc else ""
        print_fn(
            f"    RUL: {data['rul']:.1f}{mae_str}"
        )
    if "rate_multiplier" in data:
        r = data["rate_multiplier"]
        if r < 0.7:
            tag = "slower than fleet — healthy"
        elif r > 1.3:
            tag = "faster than fleet — concerning"
        else:
            tag = "normal"
        print_fn(f"    Degradation: {r:.2f}x fleet avg ({tag})")
    if "risk_level" in data:
        from src.models.what_if import prettify_sensors
        cause = prettify_sensors(data.get("primary_cause", ""))
        print_fn(f"    Risk: {data['risk_level']} — {cause}")
    if "anomalous" in data:
        if data["anomalous"]:
            from src.models.what_if import prettify_sensors as _ps
            print_fn(f"    Anomalies: {_ps(', '.join(data['flagged_sensors']))}")
        else:
            print_fn("    No anomalies detected")
    if "urgency" in data:
        rul_val = data.get("rul", "?")
        print_fn(f"    Maintenance: {data['urgency']} (RUL {rul_val})")
    if "best_condition" in data:
        best = data["best_condition"]
        worst = data["worst_condition"]
        projs = data.get("projections", [])
        if projs:
            best_p = next((p for p in projs if p["condition"] == best), projs[0])
            worst_p = next((p for p in projs if p["condition"] == worst), projs[-1])
            best_label = best_p.get("label", f"cond {best}")
            worst_label = worst_p.get("label", f"cond {worst}")
            print_fn(
                f"    Best: {best_label} | Worst: {worst_label} | "
                f"Spread: {data.get('spread', 0):.1f} RUL"
            )
        else:
            print_fn(
                f"    Best condition: {best}, "
                f"Spread: {data.get('spread', 0):.1f} RUL"
            )
    if "stress_category" in data:
        print_fn(
            f"    Stress: {data['stress_category']} "
            f"({data.get('pct_high_stress', 0) * 100:.0f}% high-stress)"
        )
    if "candidates" in data:
        print_fn(f"    {len(data['candidates'])} candidates found")
    if "feasible" in data and "legs" in data:
        status = "feasible" if data["feasible"] else "NOT FEASIBLE"
        safe_tag = " (safe)" if data.get("safe") else ""
        parts = []
        for lg in data["legs"]:
            lbl = lg.get("label", "C" + str(lg["condition"]))
            parts.append(f"{lbl} ({lg['cycles']}c)")
        legs_desc = " → ".join(parts)
        print_fn(f"    Journey: {legs_desc}")
        print_fn(f"    Final RUL: {data.get('final_rul', '?'):.1f} — {status}{safe_tag}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
# Template-based recommendation builder
# ---------------------------------------------------------------------------

def _build_recommendation(
    engine_id: Optional[int],
    rul: float,
    rate: float,
    accuracy: dict,
    diag_data: dict,
    maint_data: dict,
    cond_data: dict,
    repl_data: dict,
    journey_results: list,
    stress_data: dict,
) -> str:
    """Build a deterministic recommendation from tool data — no LLM."""
    from src.models.what_if import (
        CONDITION_LABELS, prettify_sensors,
        RUL_HARD_FLOOR, RUL_SOFT_FLOOR,
    )

    if engine_id is None:
        return "No engine specified."

    lines: list = []

    if rul <= RUL_HARD_FLOOR:
        # ── GROUNDED ──
        lines.append("IMMEDIATE ACTION")
        lines.append(
            f"Engine {engine_id} is grounded (RUL {rul:.1f}). "
            "No flights permitted."
        )
        lines.append(
            "Remove from service for immediate maintenance."
        )
        if diag_data:
            cause = prettify_sensors(
                diag_data.get("primary_cause", ""))
            risk = diag_data.get("risk_level", "")
            if cause:
                lines.append("\nROOT CAUSE")
                lines.append(f"{cause} ({risk} risk).")
                findings = diag_data.get("sensor_findings", [])
                shown = 0
                for f in findings:
                    if f.get("category") in (
                        "accelerated_wear", "malfunction", "anomalous"
                    ) and shown < 3:
                        sensor = prettify_sensors(f["sensor"])
                        ratio = f.get("slope_ratio", 0)
                        cat = f["category"].replace("_", " ")
                        lines.append(
                            f"  - {sensor}: {ratio:.1f}x fleet rate "
                            f"({cat})"
                        )
                        shown += 1

        lines.append("\nMAINTENANCE")
        lines.append(
            "Full inspection and service required before "
            "return to service."
        )
        cands = repl_data.get("candidates", [])
        if cands:
            c = cands[0]
            label = c.get("dominant_condition_label", "")
            match_note = (
                "compatible" if c.get("condition_match")
                else "different operating profile"
            )
            lines.append("\nREPLACEMENT")
            lines.append(
                f"Engine {c['engine_id']} "
                f"(RUL {c['predicted_rul']:.0f}, {label} — "
                f"{match_note})."
            )

    elif rul <= RUL_SOFT_FLOOR:
        # ── ADVISORY ──
        lines.append("ADVISORY")

        dominant_cond = stress_data.get("dominant_condition")
        dominant_label = None
        if dominant_cond is not None:
            dominant_label = CONDITION_LABELS.get(
                dominant_cond, f"Condition {dominant_cond}")

        lines.append(
            f"Engine {engine_id} is approaching end of serviceable "
            f"life (RUL {rul:.1f})."
        )
        lines.append(
            "Flights not recommended without operational necessity."
        )
        # Check if any one-way flight is feasible from journey data
        for jd in journey_results:
            legs = jd.get("legs", [])
            if len(legs) == 1:
                final = jd.get("final_rul", 0)
                label = legs[0].get("label", "")
                if final > RUL_HARD_FLOOR:
                    if dominant_label and label == dominant_label:
                        lines.append(
                            f"If required: one-way at {label} "
                            f"(RUL after: {final:.1f}) — this is "
                            f"the engine's primary route."
                        )
                    else:
                        lines.append(
                            f"If required: one-way at {label} "
                            f"(RUL after: {final:.1f})."
                        )
                break
        lines.append("\nACTION")
        lines.append("Schedule maintenance immediately.")
        cands = repl_data.get("candidates", [])
        if cands:
            c = cands[0]
            label = c.get("dominant_condition_label", "")
            lines.append(
                f"Replacement available: Engine {c['engine_id']} "
                f"(RUL {c['predicted_rul']:.0f}, {label})."
            )

    elif rul <= 50:
        # ── PLANNABLE RANGE ──
        lines.append("OPERATIONAL ADVISORY")

        # Lead with engine's actual operating profile
        dominant_cond = stress_data.get("dominant_condition")
        dominant_label = None
        if dominant_cond is not None:
            dominant_label = CONDITION_LABELS.get(
                dominant_cond, f"Condition {dominant_cond}")
            breakdown = stress_data.get("condition_breakdown", {})
            total_cyc = stress_data.get("total_cycles", 1)
            # Handle JSON round-trip: keys may be strings
            dom_count = (breakdown.get(dominant_cond, 0)
                         or breakdown.get(str(dominant_cond), 0))
            dom_pct = dom_count / total_cyc * 100
            lines.append(
                f"Engine {engine_id} primarily operates at "
                f"{dominant_label} ({dom_pct:.0f}% of flights)."
            )

        # Show journey results for dominant condition
        dominant_journeys = []
        other_journeys = []
        for jd in journey_results:
            legs = jd.get("legs", [])
            if legs and all(
                lg.get("condition") == dominant_cond for lg in legs
            ):
                dominant_journeys.append(jd)
            else:
                other_journeys.append(jd)

        if dominant_journeys:
            lines.append("")
            for jd in dominant_journeys:
                legs = jd.get("legs", [])
                n_legs = len(legs)
                trip_type = "Round trip" if n_legs >= 2 else "One-way"
                final = jd.get("final_rul", 0)
                consumed = jd.get("rul_consumed", 0)
                if final > RUL_SOFT_FLOOR:
                    status = "safe"
                elif final > RUL_HARD_FLOOR:
                    status = "at risk"
                else:
                    status = "unsafe"
                lines.append(
                    f"  {trip_type} at {dominant_label}: "
                    f"RUL {final:.1f} after ({status}, "
                    f"consumes {consumed:.1f})"
                )

        # Condition summary — use journey results to determine safety
        # for short trips, not just the 10-cycle comparison horizon
        journey_safe_conds = set()
        for jd in journey_results:
            if jd.get("feasible", False):
                for lg in jd.get("legs", []):
                    journey_safe_conds.add(lg.get("condition"))

        # If both best and worst conditions are journey-safe, all
        # intermediate conditions are also safe (monotonic cost ordering)
        best_tested = cond_data.get("best_condition") in journey_safe_conds
        worst_tested = cond_data.get("worst_condition") in journey_safe_conds
        all_implicitly_safe = best_tested and worst_tested

        safe_conds = []
        unsafe_conds = []
        for p in cond_data.get("projections", []):
            cond_id = p.get("condition")
            label = p.get("label", f"Condition {cond_id}")
            cost = p.get("rul_cost_per_cycle", 0)
            if (p.get("safe_for_journey")
                    or cond_id in journey_safe_conds
                    or all_implicitly_safe):
                safe_conds.append((label, cost))
            else:
                unsafe_conds.append(label)

        if safe_conds:
            cond_strs = [f"{lbl} ({c:.2f}/flight)" for lbl, c in safe_conds]
            lines.append(f"\nCleared for: {', '.join(cond_strs)}.")
        if unsafe_conds:
            lines.append(
                f"Not advised: {', '.join(unsafe_conds)}.")

        best = cond_data.get("best_condition")
        if best is not None:
            best_label = CONDITION_LABELS.get(
                best, f"Condition {best}")
            lines.append(f"Preferred routing: {best_label}.")

        # Degradation note
        if rate > 1.3:
            pct = (rate - 1) * 100
            cause = ""
            if diag_data:
                cause = prettify_sensors(
                    diag_data.get("primary_cause", ""))
            if cause:
                lines.append("\nDEGRADATION CONCERN")
                lines.append(
                    f"Degrading {pct:.0f}% faster than fleet "
                    f"average — {cause}."
                )
            else:
                lines.append(
                    f"\nDegradation: {pct:.0f}% above fleet "
                    f"average — monitor closely."
                )
        elif rate < 0.7:
            lines.append(
                "\nDegradation: Below fleet average — healthy.")

        # Maintenance — don't contradict journey feasibility
        if maint_data:
            urg = maint_data.get("urgency", "")
            within = maint_data.get(
                "suggested_within_cycles", "")

            # Check if any journey at dominant condition is feasible
            dominant_safe = any(
                jd.get("final_rul", 0) > RUL_SOFT_FLOOR
                for jd in dominant_journeys
            )
            any_safe = any(
                jd.get("feasible", False) for jd in journey_results
            )

            if urg == "IMMEDIATE" and (dominant_safe or any_safe):
                # Don't say "ground" when the engine can still fly safely
                lines.append("\nMAINTENANCE: HIGH")
                lines.append(
                    f"Service required within {within} flights. "
                    f"Engine can continue at cleared conditions "
                    f"until maintenance is scheduled."
                )
            elif urg == "IMMEDIATE":
                lines.append(f"\nMAINTENANCE: {urg}")
                lines.append(
                    "Ground for service. No safe operating "
                    "conditions remain at this RUL."
                )
            elif urg == "HIGH":
                lines.append(f"\nMAINTENANCE: {urg}")
                lines.append(
                    f"Schedule service within {within} flights "
                    f"to remain above the recommended threshold (RUL {RUL_SOFT_FLOOR})."
                )
            else:
                lines.append(f"\nMAINTENANCE: {urg}")
                lines.append(
                    f"Schedule service within {within} flights."
                )

    else:
        # ── HEALTHY (RUL > 50) ──
        mae = accuracy.get("mae", 15)
        lines.append("CLEARED")
        lines.append(
            f"Engine {engine_id} is healthy and cleared for "
            f"all operations."
        )
        if rate > 1.3:
            pct = (rate - 1) * 100
            lines.append(
                f"Note: Degradation {pct:.0f}% above fleet "
                f"average — monitor closely."
            )
        elif rate < 0.7:
            lines.append(
                "Degradation below fleet average — "
                "minimal wear."
            )
        else:
            lines.append(
                f"Degradation within normal range "
                f"({rate:.2f}x fleet)."
            )
        lines.append(
            f"Model accuracy at this range: "
            f"+/-{mae:.0f} RUL."
        )
        lines.append(
            "Next assessment: when RUL approaches 50."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """ReAct agent using local Llama 3.1 8B via ChatHuggingFace."""

    def __init__(self, print_fn: Callable[..., None] = print):
        self.print_fn = print_fn
        self._chat: Any = None

    def load(self) -> None:
        _get_tool_map()  # pre-import tools
        self.print_fn("  Tools loaded.")

    def run(
        self,
        engine_id: Optional[int] = None,
        query_text: str = "",
    ) -> Dict[str, Any]:

        t0 = time.time()

        if not query_text and engine_id is not None:
            query_text = (
                f"Perform a complete health analysis of engine {engine_id}. "
                f"Follow the analytical protocol."
            )
        elif engine_id is not None and str(engine_id) not in query_text:
            query_text = f"Engine {engine_id}: {query_text}"

        tool_calls: List[Dict] = []
        error: Optional[str] = None
        tools = _get_tool_map()

        def _call(tname: str, tparams: dict) -> Optional[dict]:
            """Call a tool, record it, print summary, return parsed output."""
            self.print_fn(f"\n  -> {tname}({json.dumps(tparams)})")
            try:
                result_str = tools[tname].invoke(tparams)
                _print_tool_summary(result_str, self.print_fn)
                tool_calls.append({
                    "tool": tname,
                    "input": json.dumps(tparams),
                    "output": result_str,
                })
                return json.loads(result_str)
            except Exception as exc:
                self.print_fn(f"    Error: {exc}")
                return None

        # ---- Phase 1: Core assessment (always) ----
        rul_data: dict = {}
        deg_data: dict = {}
        cond_data: dict = {}

        if engine_id is not None:
            rul_data = _call("estimate_rul", {"engine_id": engine_id}) or {}
            deg_data = _call("compute_degradation_rate", {"engine_id": engine_id}) or {}

        rul = rul_data.get("rul", 999)
        rate = deg_data.get("rate_multiplier", 1.0)
        accuracy = rul_data.get("accuracy", {})

        # ---- Phase 2: Smart depth — decide further analysis ----
        diag_data: dict = {}
        maint_data: dict = {}

        if engine_id is not None:
            # Degradation assessment: only diagnose for fast degradation
            if rate > 1.3:
                self.print_fn(
                    f"\n  Thinking: Degradation rate {rate:.2f}x is faster than fleet "
                    f"— running root-cause analysis."
                )
                diag_data = _call("diagnose_engine", {"engine_id": engine_id}) or {}
            elif rate < 0.7:
                self.print_fn(
                    f"\n  Thinking: Degradation rate {rate:.2f}x is below fleet average "
                    f"— checking sensor readings for possible drift."
                )
                diag_data = _call("diagnose_engine", {"engine_id": engine_id}) or {}

            from src.models.what_if import (
                RUL_HARD_FLOOR, RUL_SOFT_FLOOR, CONDITION_LABELS,
            )

            if rul <= RUL_HARD_FLOOR:
                # ── GROUNDED: at or below safety limit ──
                self.print_fn(
                    f"\n  Thinking: RUL {rul:.0f} is at or below the safety "
                    f"limit ({RUL_HARD_FLOOR}) — engine must be grounded."
                )
                maint_data = _call("schedule_maintenance", {"engine_id": engine_id}) or {}
                self.print_fn(
                    "\n  Thinking: Engine grounded "
                    "— searching for replacements."
                )
                _call("find_replacement_engine", {
                    "min_rul": 60.0, "max_results": 3,
                    "target_engine_id": engine_id,
                })

            elif rul <= RUL_SOFT_FLOOR:
                # ── NOT ADVISED: within advisory limit ──
                cond_data = _call("compare_conditions", {
                    "engine_id": engine_id, "cycles": 10,
                }) or {}
                best_cond = cond_data.get("best_condition", 0)

                stress_data = _call("get_stress_profile", {
                    "engine_id": engine_id,
                }) or {}
                dominant_cond = stress_data.get("dominant_condition", best_cond)

                self.print_fn(
                    f"\n  Thinking: RUL {rul:.0f} is within advisory limit "
                    f"({RUL_SOFT_FLOOR}) — testing one-way flight at "
                    f"dominant condition ({CONDITION_LABELS.get(dominant_cond, dominant_cond)})."
                )
                _call("forecast_journey", {
                    "engine_id": engine_id,
                    "legs": json.dumps([
                        {"condition": dominant_cond, "cycles": 1},
                    ]),
                })

                self.print_fn(
                    f"\n  Thinking: RUL {rul:.0f} requires maintenance."
                )
                maint_data = _call("schedule_maintenance", {"engine_id": engine_id}) or {}

                all_unsafe = cond_data and all(
                    not p.get("safe_for_journey")
                    for p in cond_data.get("projections", [])
                )
                if all_unsafe:
                    self.print_fn(
                        "\n  Thinking: No safe operating conditions "
                        "— searching for replacement engines."
                    )
                    _call("find_replacement_engine", {
                        "min_rul": 60.0, "max_results": 3,
                        "target_engine_id": engine_id,
                    })

            elif rul <= 50:
                # ── PLANNABLE RANGE: detailed flight assessment ──
                cond_data = _call("compare_conditions", {
                    "engine_id": engine_id, "cycles": 10,
                }) or {}

                stress_data = _call("get_stress_profile", {
                    "engine_id": engine_id,
                }) or {}

                best_cond = cond_data.get("best_condition", 0)
                worst_cond = cond_data.get("worst_condition", 5)
                dominant_cond = stress_data.get("dominant_condition", best_cond)

                self.print_fn(
                    f"\n  Thinking: RUL {rul:.0f} is in plannable range "
                    f"— assessing round-trip flights at each relevant condition."
                )

                # Same-condition round trips (outbound + return at same condition)
                def _round_trip(c: int) -> list:
                    return [{"condition": c, "cycles": 1},
                            {"condition": c, "cycles": 1}]

                # Test dominant, best, and worst conditions (deduplicated)
                conds_to_test = list(dict.fromkeys(
                    [dominant_cond, best_cond, worst_cond]))

                combos = [_round_trip(c) for c in conds_to_test]
                if rul < 30:
                    combos.append([
                        {"condition": best_cond, "cycles": 1},
                    ])

                for combo in combos:
                    _call("forecast_journey", {
                        "engine_id": engine_id,
                        "legs": json.dumps(combo),
                    })

                # Always schedule maintenance for plannable-range engines
                self.print_fn(
                    f"\n  Thinking: RUL {rul:.0f} — checking maintenance schedule."
                )
                maint_data = _call("schedule_maintenance", {"engine_id": engine_id}) or {}

                # Check if engine is truly unsafe — consider journey
                # results (short trips) not just 10-cycle comparisons
                all_10c_unsafe = cond_data and all(
                    not p.get("safe_for_journey")
                    for p in cond_data.get("projections", [])
                )
                journey_any_feasible = False
                for tc in tool_calls:
                    if tc["tool"] == "forecast_journey":
                        try:
                            jd = json.loads(tc["output"])
                            if jd.get("feasible", False):
                                journey_any_feasible = True
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
                if all_10c_unsafe and not journey_any_feasible:
                    self.print_fn(
                        "\n  Thinking: No safe operating conditions "
                        "— searching for replacement engines."
                    )
                    _call("find_replacement_engine", {
                        "min_rul": 60.0, "max_results": 3,
                        "target_engine_id": engine_id,
                    })
            else:
                # RUL > 50: model uncertainty too high for detailed planning
                mae = accuracy.get("mae", 15)
                self.print_fn(
                    f"\n  Thinking: RUL {rul:.0f} is well above service threshold. "
                    f"Model accuracy at this range: ±{mae:.0f} RUL (MAE). "
                    f"Detailed flight planning not meaningful at this stage."
                )

        # ---- Phase 3: Build recommendation from data ----
        repl_data: dict = {}
        journey_results: list = []
        stress_from_tools: dict = {}
        for tc in tool_calls:
            try:
                parsed = json.loads(tc["output"])
                if tc["tool"] == "find_replacement_engine":
                    repl_data = parsed
                elif tc["tool"] == "forecast_journey":
                    journey_results.append(parsed)
                elif tc["tool"] == "get_stress_profile":
                    stress_from_tools = parsed
            except (json.JSONDecodeError, TypeError):
                continue

        final_answer: Optional[str] = _build_recommendation(
            engine_id=engine_id,
            rul=rul,
            rate=rate,
            accuracy=accuracy,
            diag_data=diag_data,
            maint_data=maint_data,
            cond_data=cond_data,
            repl_data=repl_data,
            journey_results=journey_results,
            stress_data=stress_from_tools,
        )

        latency_ms = int((time.time() - t0) * 1000)

        output: Dict[str, Any] = {
            "engine_id": engine_id,
            "query": query_text,
            "recommendation": final_answer or "Agent did not produce a recommendation.",
            "tool_calls": tool_calls,
            "turns": len(tool_calls),
            "latency_ms": latency_ms,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._persist(output)
        return output

    def _persist(self, result: Dict) -> None:
        try:
            with open(_RUNS_FP, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, default=str) + "\n")
        except OSError:
            pass
