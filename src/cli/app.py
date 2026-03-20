"""Engine Health Agent — interactive CLI.

Uses a deterministic protocol orchestrator + ForecastNet ensemble to analyse
turbofan engine health on the NASA CMAPSS FD004 dataset.

Usage:
    python -m src.cli.app              # Interactive mode
    python -m src.cli.app --engine 42  # Analyse engine 42
    python -m src.cli.app --demo 1     # Run demo scenario
    python -m src.cli.app --fleet      # Fleet health overview (all 248 engines)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import warnings
from typing import Optional

# Ensure project root is on the path
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

warnings.filterwarnings("ignore")


# -- Pretty printing --------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
W = 72


def banner(text: str) -> None:
    print(f"\n{BOLD}{'=' * W}")
    print(f"  {text}")
    print(f"{'=' * W}{RESET}")


def section(text: str) -> None:
    print(f"\n{CYAN}{'-' * W}")
    print(f"  {text}")
    print(f"{'-' * W}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}+{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def err(msg: str) -> None:
    print(f"  {RED}x{RESET} {msg}")


def wrap_print(text: str, indent: int = 4) -> None:
    for line in textwrap.wrap(str(text), width=W - indent):
        print(f"{' ' * indent}{line}")


# -- Analysis display -------------------------------------------------------


def _extract_tool_data(result: dict) -> dict:
    """Extract parsed tool results from the agent output."""
    data: dict = {}
    for tc in result.get("tool_calls", []):
        try:
            parsed = __import__("json").loads(tc.get("output", "{}"))
            key = tc["tool"]
            if key == "forecast_journey":
                data.setdefault("forecast_journeys", []).append(parsed)
            else:
                data[key] = parsed
        except (ValueError, TypeError):
            pass
    return data


def _display_structured(result: dict) -> None:
    """Build structured display from real tool data + LLM recommendation."""
    from src.models.what_if import prettify_sensors

    engine_id = result.get("engine_id")
    tools = _extract_tool_data(result)

    rul_data = tools.get("estimate_rul", {})
    deg_data = tools.get("compute_degradation_rate", {})
    diag_data = tools.get("diagnose_engine", {})
    maint_data = tools.get("schedule_maintenance", {})
    journeys = tools.get("forecast_journeys", [])
    replacement_data = tools.get("find_replacement_engine", {})

    # -- Engine status with model accuracy (not ensemble CI)
    if rul_data:
        rul = float(rul_data.get("rul", 0))
        acc = rul_data.get("accuracy", {})
        label = f"ENGINE {engine_id}" if engine_id else "ENGINE"
        mae_str = f"(±{acc['mae']:.0f} RUL)" if acc else ""
        print(f"  {BOLD}{label}: RUL {rul:.1f} {mae_str}{RESET}")

        # Bias note when model has significant bias in this range
        if acc and abs(acc.get("bias", 0)) > 3:
            direction = "over" if acc["bias"] > 0 else "under"
            bc_rul = rul - acc["bias"]
            print(f"  {DIM}Model tends to {direction}-predict by "
                  f"~{abs(acc['bias']):.0f} RUL here. "
                  f"Bias-corrected: ~{bc_rul:.0f}{RESET}")

    # -- Degradation with sensor-level detail
    if deg_data:
        from src.models.what_if import SENSOR_LABELS
        rate = deg_data.get("rate_multiplier", 1.0)
        if rate < 0.7:
            s_slopes = deg_data.get("sensor_slopes", {})
            f_slopes = deg_data.get("fleet_slopes", {})
            below = [
                prettify_sensors(s)
                for s, v in s_slopes.items()
                if abs(v) < abs(f_slopes.get(s, v or 1)) * 0.6
                and abs(f_slopes.get(s, 1)) > 1e-6
            ]
            sensor_note = (
                f" — {', '.join(below[:2])} reading below fleet norm; "
                f"possible sensor calibration drift"
                if below else ""
            )
            print(f"  Degradation: {rate:.2f}x fleet average — "
                  f"{GREEN}below normal{RESET}{sensor_note}")
        elif rate > 1.3:
            pct = (rate - 1) * 100
            print(f"  Degradation: {rate:.2f}x fleet average — "
                  f"{RED}abnormal{RESET}")
            print(f"    Degrading {pct:.0f}% faster than fleet average.")
            if diag_data:
                findings = diag_data.get("sensor_findings", [])
                shown = 0
                for finding in findings:
                    if finding.get("category") in (
                        "accelerated_wear", "malfunction", "anomalous"
                    ) and shown < 3:
                        sensor_name = prettify_sensors(finding["sensor"])
                        ratio = finding.get("slope_ratio", 0)
                        cat = finding["category"].replace("_", " ")
                        print(f"    • {sensor_name}: "
                              f"{ratio:.1f}× fleet rate — {cat}")
                        shown += 1
            else:
                s_slopes = deg_data.get("sensor_slopes", {})
                f_slopes = deg_data.get("fleet_slopes", {})
                for s, v in s_slopes.items():
                    fleet_v = f_slopes.get(s, 0)
                    if abs(fleet_v) > 1e-6:
                        ratio = abs(v) / abs(fleet_v)
                        if ratio > 1.3:
                            sensor_name = SENSOR_LABELS.get(s, s)
                            print(f"    • {sensor_name}: "
                                  f"{ratio:.1f}× fleet rate")
        else:
            print(f"  Degradation: {rate:.2f}x fleet average — "
                  f"{GREEN}normal{RESET}")

    # -- Diagnosis
    if diag_data:
        cause = prettify_sensors(
            diag_data.get("primary_cause", "standard operational wear")
        )
        risk = diag_data.get("risk_level", "")
        colour = RED if risk == "CRITICAL" else (
            YELLOW if risk == "HIGH" else GREEN)
        if deg_data and deg_data.get("rate_multiplier", 1.0) > 1.3:
            print(f"  {colour}Risk: {risk}{RESET} — {cause}")
        else:
            print(f"  Diagnosis: {cause} ({colour}risk: {risk}{RESET})")

    # -- Flight assessment
    rul = float(rul_data.get("rul", 0)) if rul_data else 999
    acc = rul_data.get("accuracy", {}) if rul_data else {}
    mae = acc.get("mae", 4) if acc else 4

    if rul <= 6:
        # GROUNDED — at or below safety limit
        print(f"\n  {BOLD}FLIGHT STATUS{RESET}")
        print(f"    {RED}✗ GROUNDED{RESET} — RUL {rul:.1f}. "
              f"High risk of engine failure.")
        print("    No flights permitted. Remove from service "
              "for immediate maintenance.")
    elif rul <= 10:
        # NOT ADVISED — within advisory limit
        print(f"\n  {BOLD}FLIGHT STATUS{RESET}")
        print(f"    {YELLOW}⚠ NOT ADVISED{RESET} — RUL {rul:.1f}. "
              f"Approaching end of serviceable life.")
        print("    Flights not recommended without "
              "operational necessity. Schedule maintenance now.")
        if journeys:
            _display_journeys(journeys, mae)
    elif rul <= 50:
        # PLANNABLE RANGE — show flight assessment
        print(f"\n  {BOLD}FLIGHT ASSESSMENT{RESET} "
              f"(model accuracy: ±{mae:.0f} RUL at this range)")
        if journeys:
            _display_journeys(journeys, mae)
    else:
        # HEALTHY — model uncertainty too high for detailed planning
        mae = acc.get("mae", 15)
        print(f"\n  {BOLD}FLIGHT STATUS{RESET}")
        print(f"    {GREEN}✓ CLEARED{RESET} — All operating conditions "
              f"permitted.")
        print(f"    Model accuracy at this lifecycle stage: "
              f"±{mae:.0f} RUL (MAE).")
        print(f"    {DIM}Detailed journey assessment available when "
              f"RUL approaches 50.{RESET}")

    # -- Maintenance
    if maint_data:
        urgency = maint_data.get("urgency", "?")
        colour = RED if urgency == "IMMEDIATE" else (
            YELLOW if urgency == "HIGH" else GREEN)
        print(f"\n  {BOLD}MAINTENANCE{RESET}: {colour}{urgency}{RESET}")
        rul_now = maint_data.get("rul", "?")
        urgency = maint_data.get("urgency", "")
        if urgency == "IMMEDIATE":
            print(f"    RUL {rul_now} — ground immediately due to high "
                  f"risk of engine failure.")
            print(f"    {DIM}Safety threshold: 6 RUL. Full inspection "
                  f"and service required before any further "
                  f"operation.{RESET}")
        elif urgency == "HIGH":
            print(f"    RUL {rul_now} — schedule service now to remain "
                  f"above the recommended threshold (RUL 10).")
        else:
            print(f"    RUL {rul_now} — routine service recommended "
                  f"to maintain operational readiness.")

    # -- Replacement engines
    if replacement_data and replacement_data.get("candidates"):
        cands = replacement_data["candidates"]
        print(f"\n  {BOLD}REPLACEMENT CANDIDATES{RESET}")
        for c in cands:
            cond_info = ""
            if c.get("dominant_condition_label"):
                match_tag = ""
                if c.get("condition_match"):
                    match_tag = f" {GREEN}✓ compatible{RESET}"
                elif c.get("condition_match") is False:
                    match_tag = f" {YELLOW}~ different profile{RESET}"
                cond_info = (f" | Primary: {c['dominant_condition_label']}"
                            f"{match_tag}")
            print(f"    Engine {c['engine_id']}: "
                  f"RUL {c['predicted_rul']:.0f}{cond_info}")


def _rul_tag(rul_val: float, mae: float) -> tuple:
    """Return (status_text, colour) for a RUL value."""
    if rul_val <= 6:
        return "✗ grounded", RED
    elif rul_val <= 10:
        return "⚠ not advised", YELLOW
    elif rul_val - mae <= 6:
        return "⚠ risk zone", YELLOW
    else:
        return "✓ safe", GREEN


def _journey_verdict(final_rul: float, mae: float) -> tuple:
    """Return (verdict_text, colour) for overall journey outcome."""
    if final_rul <= 6:
        return "✗ NOT SAFE", RED
    elif final_rul <= 10:
        return "⚠ NOT ADVISED", YELLOW
    elif final_rul - mae <= 6:
        return "⚠ CAUTION", YELLOW
    else:
        return "✓ SAFE", GREEN



def _display_journeys(journeys: list, mae: float) -> None:
    """Display journey forecasts as outbound/return narratives."""
    print(f"\n    {BOLD}Journey assessment:{RESET}")
    for j in journeys:
        if j.get("error"):
            continue
        legs = j.get("legs", [])
        final_rul = j.get("final_rul", 0)

        # Detect journey structure (new format: no ground runs)
        is_round_trip = len(legs) == 2
        is_one_way = len(legs) == 1

        if is_round_trip:
            out_leg = legs[0]
            ret_leg = legs[1]
            out_label = out_leg.get("label",
                                   f"Condition {out_leg['condition']}")
            ret_label = ret_leg.get("label",
                                   f"Condition {ret_leg['condition']}")
            out_rul = out_leg.get("rul_after", 0)

            # Ensure non-monotonic artifacts don't confuse display
            display_out = out_rul
            display_ret = min(final_rul, out_rul)

            if out_label == ret_label:
                title = f"Round trip at {out_label}"
            else:
                title = (f"Outbound {out_label}, "
                         f"return {ret_label}")

            verdict, v_colour = _journey_verdict(display_ret, mae)
            print(f"\n      {v_colour}{verdict}{RESET}  {title}")

            out_tag, out_c = _rul_tag(display_out, mae)
            ret_tag, ret_c = _rul_tag(display_ret, mae)

            print(f"        After outbound: RUL ≈ {display_out:.1f}  "
                  f"{out_c}{out_tag}{RESET}")
            print(f"        After return:   RUL ≈ {display_ret:.1f}  "
                  f"{ret_c}{ret_tag}{RESET}")

            # Decision narrative for split outcomes
            if display_ret <= 6 and display_out > 10:
                print(f"        {DIM}Outbound is safe but the return "
                      f"would bring RUL to ≈{display_ret:.0f} —")
                print(f"        within ±{mae:.0f} RUL of engine "
                      f"failure. Return flight not recommended."
                      f"{RESET}")
            elif display_ret <= 10 and display_out > 10:
                print(f"        {DIM}Outbound is safe. Return brings "
                      f"RUL into the advisory zone —")
                print(f"        not recommended without "
                      f"contingency plan.{RESET}")

        elif is_one_way:
            cruise = legs[0]
            cruise_label = cruise.get(
                "label", f"Condition {cruise['condition']}")

            verdict, v_colour = _journey_verdict(final_rul, mae)
            tag, t_colour = _rul_tag(final_rul, mae)
            print(f"\n      {v_colour}{verdict}{RESET}  "
                  f"One-way at {cruise_label}")
            print(f"        After flight: RUL ≈ {final_rul:.1f}  "
                  f"{t_colour}{tag}{RESET}")

        else:
            # Generic fallback (legacy 4-leg format or other)
            parts = []
            for lg in legs:
                lbl = lg.get("label", f"C{lg['condition']}")
                parts.append(f"{lbl} ({lg['cycles']}c)")
            verdict, v_colour = _journey_verdict(final_rul, mae)
            print(f"\n      {v_colour}{verdict}{RESET}  "
                  f"{' → '.join(parts)}")
            print(f"        Final RUL: ≈{final_rul:.0f}")


def display_drift_report(data: dict) -> None:
    """Print PSI drift report."""
    from src.models.what_if import prettify_sensors

    if data.get("error"):
        err(data["error"])
        return

    banner("Sensor Drift Report — NASA CMAPSS FD004")
    ts = data.get("timestamp", "")[:19].replace("T", " ")
    total = data.get("features_checked", 0)
    sig = data.get("significant_drift", 0)
    mod = data.get("moderate_drift", 0)
    ok_n = data.get("no_drift", 0)

    print(f"\n  {DIM}Computed: {ts} | {total} features checked{RESET}")
    print(f"\n  {GREEN}✓ No drift{RESET}       : {ok_n} features  (PSI < 0.10)")
    print(f"  {YELLOW}⚠ Moderate drift{RESET} : {mod} features  (PSI 0.10–0.25)")
    print(f"  {RED}✗ Significant{RESET}    : {sig} features  (PSI > 0.25)")

    details = data.get("details", [])
    if details:
        print(f"\n  {'Feature':<14} {'PSI':>7}  {'Status'}")
        print(f"  {'-'*40}")
        for row in details:
            feat = row.get("feature", "")
            psi = row.get("psi_score", 0.0)
            level = row.get("drift_level", "none")
            label = prettify_sensors(feat) if feat.startswith("n_s") else feat
            if level == "significant":
                colour, symbol = RED, "✗"
            elif level == "moderate":
                colour, symbol = YELLOW, "⚠"
            else:
                colour, symbol = GREEN, "✓"
            print(f"  {label:<14} {psi:>7.4f}  {colour}{symbol} {level}{RESET}")

    if sig > 0:
        flagged = ", ".join(data.get("flagged", [])[:5])
        print(f"\n  {RED}ACTION: Significant drift in {flagged}.")
        print("  Investigate whether operating conditions or sensor calibration")
        print(f"  have changed. Model predictions may be less reliable.{RESET}")
    elif mod > 0:
        print(f"\n  {YELLOW}MONITOR: Moderate drift detected. Continue monitoring.{RESET}")
    else:
        print(f"\n  {GREEN}All features within normal distribution range.{RESET}")

    print(f"\n  {DIM}PSI algorithm: Population Stability Index")
    print(f"  Baseline: FD004 training set | Current: FD004 test set{RESET}")
    print(f"{'=' * W}\n")


def display_fleet_health(data: dict) -> None:
    """Print fleet health overview bucketed by RUL tier."""
    counts = data["counts"]
    total = data["total"]
    buckets = data["buckets"]

    banner(f"Fleet Health Overview — NASA CMAPSS FD004 ({total} engines)")

    tier_cfg = [
        ("GROUNDED", f"{RED}✗ GROUNDED{RESET}", "RUL ≤ 6"),
        ("CRITICAL", f"{RED}⚠ CRITICAL{RESET}", "RUL 7–10"),
        ("ADVISORY", f"{YELLOW}⚠ ADVISORY{RESET}", "RUL 11–30"),
        ("HEALTHY",  f"{GREEN}✓ HEALTHY{RESET}",  "RUL > 30"),
    ]

    for tier, label, range_str in tier_cfg:
        n = counts[tier]
        entries = buckets[tier]
        print(f"\n  {label}  ({range_str})  —  {BOLD}{n}{RESET} engines")

        if tier == "HEALTHY":
            print(f"    {DIM}Not listed individually — no immediate action required.{RESET}")
            continue

        if n == 0:
            print(f"    {DIM}None.{RESET}")
            continue

        for e in entries:
            eid = e["engine_id"]
            rul = e["rul"]
            rate = e.get("rate_multiplier")
            rate_str = ""
            if rate is not None and rate > 1.3:
                rate_str = f"  {YELLOW}[abnormal degradation ×{rate:.1f}]{RESET}"
            print(f"    Engine {eid:>3}: RUL {rul:>3}{rate_str}")

    grounded = counts["GROUNDED"]
    critical = counts["CRITICAL"]
    advisory = counts["ADVISORY"]
    healthy  = counts["HEALTHY"]
    print(f"\n  {DIM}Summary: {grounded} grounded, {critical} critical, "
          f"{advisory} advisory, {healthy} healthy{RESET}")
    print(f"  {DIM}Source: ground-truth RUL (RUL_FD004.txt) — no model cap applied{RESET}")
    print(f"{'=' * W}\n")


def run_analysis(orchestrator, engine_id: Optional[int] = None,
                 query: str = "") -> None:
    """Run the agent on an engine (or fleet query) and display results."""
    if engine_id is not None:
        banner(f"Analysing Engine {engine_id}")
    else:
        banner("Fleet Analysis")

    print(f"\n{DIM}  Agent reasoning ...{RESET}")

    result = orchestrator.run(engine_id=engine_id, query_text=query)

    if result.get("error"):
        err(f"Agent error: {result['error']}")
        if result.get("recommendation"):
            section("Partial Analysis")
            print()
            wrap_print(result["recommendation"])
        return

    section("Report")

    # Structured data section from real tool outputs
    _display_structured(result)

    # LLM recommendation
    rec = result.get("recommendation", "")
    if rec and rec != "Agent did not produce a recommendation.":
        section("Recommendation")
        print()
        for line in rec.split("\n"):
            stripped = line.strip()
            if not stripped:
                print()
                continue
            elif stripped.startswith(("- ", "• ", "* ")):
                print(f"    {GREEN}{stripped}{RESET}")
            elif re.match(r'^\d+[\.)\:]\s', stripped):
                print(f"    {BOLD}{stripped}{RESET}")
            elif stripped.isupper() or stripped.endswith(":"):
                print(f"\n    {BOLD}{stripped}{RESET}")
            else:
                wrap_print(stripped)
        print()

    # footer
    latency = result.get("latency_ms", 0) / 1000
    n_tools = result.get("turns", 0)
    print(f"  {DIM}Tools called: {n_tools} | Time: {latency:.1f}s{RESET}")
    print(f"{'=' * W}\n")


# -- Demo scenarios ----------------------------------------------------------

DEMOS = [
    {
        "id": 1,
        "title": "Critical engine — analyse engine 12",
        "description": (
            "Engine 12 has only ~7 cycles of true RUL remaining. "
            "The agent should detect abnormal degradation and recommend "
            "grounding."
        ),
        "engine_id": 12,
        "query": (
            "Perform a complete health analysis of engine 12. "
            "Follow the analytical protocol."
        ),
    },
    {
        "id": 2,
        "title": "Route planning — which conditions suit engine 42?",
        "description": (
            "Engine 42 has ~30 cycles of RUL. The agent should assess "
            "degradation normality and compare operating conditions."
        ),
        "engine_id": 42,
        "query": (
            "Analyse engine 42. Determine whether degradation is normal "
            "and which operating conditions are best for it."
        ),
    },
    {
        "id": 3,
        "title": "Mission assignment — best engine for 50-cycle high-stress flight",
        "description": (
            "Find the best engine for a demanding 50-cycle mission at "
            "condition 4 (high stress)."
        ),
        "engine_id": None,
        "query": (
            "Find the best engine for a 50-cycle mission at condition 4 "
            "(high stress). The engine must retain at least 30 cycles of "
            "RUL after the mission. Use find_best_engine_for_mission."
        ),
    },
    {
        "id": 4,
        "title": "Healthy engine check — engine 200",
        "description": (
            "Engine 200 likely has high RUL. The agent should confirm "
            "normal degradation."
        ),
        "engine_id": 200,
        "query": (
            "Perform a complete health analysis of engine 200. "
            "Follow the analytical protocol."
        ),
    },
]


def run_demo(orchestrator, demo_id: int) -> None:
    demo = next((d for d in DEMOS if d["id"] == demo_id), None)
    if demo is None:
        err(f"Demo {demo_id} not found. Available: 1-{len(DEMOS)}")
        return
    banner(f"Demo {demo['id']}: {demo['title']}")
    print(f"\n  {DIM}{demo['description']}{RESET}")
    run_analysis(orchestrator, demo.get("engine_id"), demo["query"])


# -- Interactive mode --------------------------------------------------------


def interactive_mode() -> None:
    banner("Engine Health Agent — NASA CMAPSS FD004")
    print(
        f"""
  Turbofan engine health analysis powered by
  Llama 3.1 8B Instruct + ForecastNet ensemble + XGBoost.

  The agent autonomously analyses engine health,
  identifies root causes, and makes recommendations.

  {BOLD}Commands:{RESET}
    <number>     Analyse engine 1-248
    fleet        Fleet health overview (all 248 engines)
    drift        Sensor drift report (PSI vs training baseline)
    demo         Show demo scenarios
    q            Quit
"""
    )

    from src.agent.orchestrator import AgentOrchestrator

    agent = AgentOrchestrator()
    print(f"  {DIM}Loading models ...{RESET}")
    agent.load()
    ok("Agent ready.\n")

    while True:
        try:
            raw = input(f"  {BOLD}engine>{RESET} ").strip()
            # Strip ANSI escape sequences (e.g. arrow keys leaking into buffer)
            raw = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        low = raw.lower()
        if low in ("q", "quit", "exit"):
            break

        if low == "fleet":
            from src.mcp_server.server import fleet_health_summary
            display_fleet_health(fleet_health_summary())
            continue

        if low == "drift":
            from src.drift.drift_monitor import run_drift_check
            print(f"  {DIM}Computing PSI drift check ...{RESET}")
            display_drift_report(run_drift_check())
            continue

        if low == "demo":
            print(f"\n  {BOLD}Demos:{RESET}")
            for d in DEMOS:
                print(f"    [{d['id']}] {d['title']}")
            try:
                did = input("\n  Demo number: ").strip()
                run_demo(agent, int(did))
            except (ValueError, EOFError, KeyboardInterrupt):
                err("Enter a demo number.")
            continue

        # Numeric input -> engine ID
        if raw.isdigit():
            eid = int(raw)
            if 1 <= eid <= 248:
                run_analysis(agent, eid)
            else:
                err("Engine ID must be 1-248 (FD004 test set).")
            continue

        # Free-form query
        if raw:
            eid = None
            m = re.search(r"engine\s*(\d+)", raw, re.IGNORECASE)
            if m:
                eid = int(m.group(1))
            run_analysis(agent, eid, raw)

    ok("Goodbye.")


# -- Entry point -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Engine Health Agent - autonomous engine analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
        Examples:
          python -m src.cli.app              # Interactive mode
          python -m src.cli.app --engine 42  # Analyse engine 42
          python -m src.cli.app --demo 1     # Run demo 1
          python -m src.cli.app --query "Find the best engine for a 30-cycle mission"
        """
        ),
    )
    parser.add_argument(
        "--engine", type=int, metavar="ID", help="Analyse engine ID (1-248)"
    )
    parser.add_argument(
        "--demo", type=int, metavar="N", help="Run demo scenario N"
    )
    parser.add_argument(
        "--query", type=str, metavar="TEXT", help="Free-form query"
    )
    parser.add_argument(
        "--fleet", action="store_true", help="Print fleet health overview (all 248 engines)"
    )
    parser.add_argument(
        "--drift", action="store_true", help="Run sensor drift report (PSI vs training baseline)"
    )

    args = parser.parse_args()

    if args.drift:
        from src.drift.drift_monitor import run_drift_check
        print(f"  {DIM}Computing PSI drift check ...{RESET}")
        display_drift_report(run_drift_check())
        return

    if args.fleet:
        from src.mcp_server.server import fleet_health_summary
        display_fleet_health(fleet_health_summary())
        return

    if args.demo is not None or args.engine is not None or args.query:
        from src.agent.orchestrator import AgentOrchestrator

        agent = AgentOrchestrator()
        agent.load()

        if args.demo is not None:
            run_demo(agent, args.demo)
        elif args.engine is not None:
            run_analysis(agent, args.engine, args.query or "")
        elif args.query:
            m = re.search(r"engine\s*(\d+)", args.query, re.IGNORECASE)
            eid = int(m.group(1)) if m else None
            run_analysis(agent, eid, args.query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
