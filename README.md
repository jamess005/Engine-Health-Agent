# Engine Health Agent — NASA CMAPSS FD004

Predictive maintenance system for turbofan engines. Given a fleet of 248 test engines
across six operating conditions, the agent predicts each engine's **Remaining Useful Life
(RUL)**, identifies which engines are safe to fly, diagnoses abnormal degradation,
and generates actionable maintenance recommendations — without an LLM in the loop.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-James%20Scott-0077B5?logo=linkedin)](https://www.linkedin.com/in/jamesscott005)

---

## Results

| Metric | Value |
|--------|-------|
| Model | ForecastNet v5c — 4-seed ensemble |
| Architecture | Multi-scale 1D CNN + Squeeze-Excitation + Multi-head Attention |
| Dataset | NASA CMAPSS FD004 (248 test engines, 6 operating conditions) |
| **RMSE** | **13.77 cycles** |
| Accuracy by lifecycle stage | ±4 RUL (early) → ±13 RUL (end-of-life) |
| RUL cap | 130 cycles (model training); uncapped for fleet ranking |

The model is evaluated on the standard NASA FD004 scoring function. RMSE of 13.77 is
achieved on the full test set of 248 engines, spanning all six operating conditions.

---

## The Problem

CMAPSS FD004 is the hardest of the four NASA datasets: 6 distinct operating conditions,
21 sensors, and run-to-failure degradation curves that vary significantly across engines.
A single RUL prediction is not enough for operational use — the fleet operator needs to
know *which engines are safe today*, *which routes each engine can handle*, and *what to
do when an engine approaches its limit*.

This system answers all three questions in a structured, auditable way.

---

## Design Decisions

### Why a deterministic orchestrator, not an LLM?

Early versions of this system used Llama 3.1 8B to reason over tool outputs and produce
recommendations. It was removed. The LLM introduced hallucinations — it would contradict
the tool data, say a route was unsafe when the numbers showed it was safe, and generate
maintenance messages that sounded plausible but were quantitatively wrong.

The deterministic orchestrator calls the same tools in the same order and builds its
recommendation from a template driven entirely by the tool outputs. Every number in the
recommendation comes from a tool call. This makes the system reliable enough to act on.

### Why ForecastNet instead of a simpler model?

Simpler regressors (gradient boosting, plain CNN) treat each operating condition as a
covariate. ForecastNet uses squeeze-excitation blocks to weight channels dynamically per
condition and multi-head attention to capture long-range degradation trends across the
cycle window. The multi-scale branches (3, 5, 7 cycle kernels) capture degradation at
different timescales simultaneously. Four random seeds are run in ensemble — this gives
a confidence interval, not just a point estimate, which the downstream tools use to
determine whether a route is safe within model uncertainty.

### Why ground-truth RUL for fleet ranking instead of model predictions?

The model caps RUL at 130 for training stability. In the test set, some engines have
true RUL up to 195. Using model predictions for fleet ranking would make the top
candidates converge on the cap — every healthy engine would look identical at RUL 130.
Fleet ranking, replacement candidate search, and the fleet health overview all use the
true RUL from `RUL_FD004.txt`, stored in the database at startup.

---

## How It Works

```
NASA CMAPSS FD004
(248 test engines, 21 sensors, 6 operating conditions)
         │
         ▼
 Condition Normalisation
 KMeans cluster (altitude × Mach × TRA) → condition label 0–5
 Per-condition z-score normalisation → comparable sensor readings
         │
         ▼
 Feature Engineering
 50-cycle sliding window → rolling slopes, means, sensor deltas
         │
         ▼
 ForecastNet v5c Ensemble (4 seeds)
 Multi-scale CNN → SE blocks → Multi-head Attention → RUL + 80% CI
         │
         ▼
 Deterministic Orchestrator
   1. estimate_rul            RUL + confidence interval + model accuracy tier
   2. compute_degradation_rate  Fleet-relative sensor slope (1.0 = average)
   3. diagnose_engine          Root-cause analysis (triggered if rate > 1.3×)
   4. compare_conditions       Project RUL under all 6 conditions
   5. forecast_journey         Per-leg RUL after outbound + return flights
   6. schedule_maintenance     Urgency tier (IMMEDIATE / HIGH / MODERATE / LOW)
   7. find_replacement_engine  Ranked by true RUL, filtered by stress category
   8. Recommendation           Template-built from tool outputs — no LLM
         │
         ▼
 SQLite DB  (data/engine_health.db)
 predictions · agent_runs · drift_metrics · raw_sensor_data · processed_features
```

---

## Example Output

```
  ENGINE 42: RUL 28.7 (±9 RUL)
  Model tends to over-predict by ~4 RUL here. Bias-corrected: ~25

  Degradation: 1.43x fleet average — abnormal
    Degrading 43% faster than fleet average.
    • Exhaust gas temperature (EGT core): 1.8× fleet rate — accelerated wear
    • Fan speed: 1.5× fleet rate — accelerated wear

  FLIGHT ASSESSMENT (model accuracy: ±9 RUL at this range)

    ✓ SAFE  Round trip at High cruise 42kft
              After outbound: RUL ≈ 27.4  ✓ safe
              After return:   RUL ≈ 26.3  ✓ safe

    ✗ NOT SAFE  Round trip at High full power 35kft
              After outbound: RUL ≈ 21.0  ✓ safe
              After return:   RUL ≈ 13.8  ✓ safe

  MAINTENANCE: HIGH
    RUL 28.7 — schedule service now to remain above the recommended threshold (RUL 10).

  REPLACEMENT CANDIDATES
    Engine 127: RUL 195 | Primary: High cruise 42kft ✓ compatible
    Engine  14: RUL 192 | Primary: High cruise 42kft ✓ compatible
    Engine  89: RUL 188 | Primary: High cruise 42kft ✓ compatible
```

The assessment is per-condition: engine 42 can safely complete a round trip at low-stress
conditions but should not attempt a high-stress route given its degradation rate. The
replacement candidates are filtered to the same stress category and sorted by true RUL.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/jamess005/Engine-Health-Agent.git
cd Engine-Health-Agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Set DB_USER to your OS username; DB_PATH is fine as default
```

### 3. Data and models

The following are excluded from version control (gitignored):

- `data/raw/` — NASA CMAPSS FD004 dataset (download from [NASA DASHlink](https://c3.ndc.nasa.gov/dashlink/resources/139/))
- `data/processed/` — preprocessed parquet files (generated by notebooks `01`–`04`)
- `models/` — trained ForecastNet checkpoint and preprocessing artefacts

Run notebooks `01` → `05` in order to reproduce all preprocessing and training steps.
The database (`data/engine_health.db`) is initialised automatically on first run — the
raw and processed data files are imported into SQLite at startup.

---

## Usage

### Interactive CLI

```bash
python -m src.cli.app
```

At the `engine>` prompt:

| Command | Action |
|---------|--------|
| `42` | Analyse engine 42 (valid range: 1–248) |
| `fleet` | Fleet overview — all 248 engines bucketed by true RUL |
| `drift` | Sensor + prediction drift report (PSI vs training baseline) |
| `demo` | Choose a built-in demo scenario |
| `q` | Quit |

### One-shot flags

```bash
python -m src.cli.app --engine 42     # Analyse a specific engine
python -m src.cli.app --fleet         # Fleet health overview
python -m src.cli.app --drift         # Run full drift check
python -m src.cli.app --demo 1        # Demo: critical engine (RUL ~7)
python -m src.cli.app --demo 2        # Demo: route planning (RUL ~30)
```

### REST API

```bash
uvicorn src.api.main:app --reload
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/agent/query` | Submit an async agent query; returns `job_id` |
| `GET` | `/agent/result/{job_id}` | Poll for completed result |
| `POST` | `/predict/rul` | Synchronous RUL prediction (no agent reasoning) |
| `GET` | `/engine/{id}/history` | Agent run history for an engine |

Agent queries run in a background thread and are persisted to the `async_jobs` and
`agent_runs` database tables — results survive server restarts.

---

## Fleet Health Overview

```bash
python -m src.cli.app --fleet
```

Buckets all 248 test engines by ground-truth RUL. Uses `RUL_FD004.txt` directly —
no model inference, runs in ~0.2s regardless of fleet size.

| Tier | Threshold | Meaning |
|------|-----------|---------|
| ✗ GROUNDED | RUL ≤ 6 | Hard safety floor — no flights permitted |
| ⚠ CRITICAL | RUL 7–10 | Maintenance before next flight |
| ⚠ ADVISORY | RUL 11–30 | Schedule maintenance — limited flights remaining |
| ✓ HEALTHY | RUL > 30 | No immediate action required |

---

## Drift Monitoring

```bash
python -m src.cli.app --drift
```

Runs **Population Stability Index (PSI)** drift checks across two dimensions:

**Input drift** — compares the distribution of each sensor feature in the test fleet
against the training baseline. Captures whether operating conditions or sensor
calibration have shifted since the model was trained.

**Output drift** — compares recent RUL predictions against the oldest predictions in
the database. Catches model decay or pipeline changes that affect prediction distributions
without necessarily triggering sensor-level alerts.

| PSI | Status |
|-----|--------|
| < 0.10 | No drift |
| 0.10–0.25 | Moderate — monitor closely |
| > 0.25 | Significant — investigate; model accuracy may be degraded |

Results are logged to the `drift_metrics` table with timestamps, enabling trend tracking
across multiple checks.

---

## Operating Conditions

FD004 engines operate across six conditions, identified by KMeans clustering on
altitude, Mach number, and throttle resolver angle (TRA):

| Cluster | Label | Stress |
|---------|-------|--------|
| 0 | High cruise 42kft | Low — efficient cruise |
| 1 | Low altitude 10kft | Low — short-haul |
| 2 | Mid economy 25kft | Low — medium cruise |
| 3 | Mid full power 20kft | High — high power at mid altitude |
| 4 | Sea-level ground run | High — ground operations |
| 5 | High full power 35kft | High — maximum stress |

Conditions 0–2 are low-stress; 3–5 are high-stress. Replacement engine search filters
by stress category to ensure the candidate's operating profile matches the engine being
replaced.

---

## Safety Thresholds

| Constant | Value | Meaning |
|----------|-------|---------|
| `RUL_HARD_FLOOR` | 6 cycles | Engine grounded — removal from service required |
| `RUL_SOFT_FLOOR` | 10 cycles | Advisory — maintenance before next flight |

Thresholds are applied across the full pipeline: journey feasibility checks, maintenance
urgency classification, and recommendation text all use these values consistently.

---

## Database

SQLite at `data/engine_health.db`, configured via `DB_PATH` in `.env`. Populated
automatically on first run from the raw and processed data files.

| Table | Contents |
|-------|----------|
| `predictions` | Per-engine RUL inference log with confidence interval and timestamp |
| `agent_runs` | Full agent execution history (engine, query, recommendation, tool calls) |
| `drift_metrics` | PSI snapshots — both sensor and prediction drift |
| `raw_sensor_data` | CMAPSS raw txt (unit, cycle, op1–3, s1–21) |
| `processed_features` | Condition-normalised features from preprocessed parquet files |
| `rul_labels` | Ground-truth RUL per test engine from `RUL_FD004.txt` |
| `async_jobs` | API job queue — persists agent query results across server restarts |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Dataset overview, sensor distributions, operating condition clustering |
| `02_feature_engineering_tree.ipynb` | Feature selection for tree-based models |
| `03_feature_engineering_forecast.ipynb` | Window construction and normalisation for ForecastNet |
| `04_tree_model.ipynb` | XGBoost baseline (reference point for ForecastNet comparison) |
| `05_forecast_model.ipynb` | ForecastNet architecture, training, ensemble results (RMSE 13.77) |
| `06_evaluation.ipynb` | Per-engine error analysis, accuracy tiers, CI calibration |

---

## Project Structure

```
src/
├── agent/
│   └── orchestrator.py      Deterministic protocol — calls tools, builds recommendation
├── api/
│   └── main.py              FastAPI endpoints; DB-backed async job store
├── cli/
│   └── app.py               Interactive CLI and one-shot flags
├── core/
│   └── logging_config.py    Centralised logging setup (reads LOG_LEVEL from .env)
├── db/
│   └── database.py          SQLite init, migration, read/write helpers
├── drift/
│   └── drift_monitor.py     PSI drift detection — sensor + prediction distributions
├── features/
│   ├── builder.py           Feature engineering (rolling stats, condition features)
│   ├── loader.py            DB-first loader with parquet fallback
│   ├── normaliser.py        Per-condition sensor normalisation
│   └── windows.py           50-cycle inference window construction
├── mcp_server/
│   └── server.py            11 MCP tools (FastMCP) with input validation
└── models/
    ├── anomaly_detector.py  Z-score anomaly detection vs healthy-engine baseline
    ├── degradation_rate.py  Fleet-relative degradation slope comparison
    ├── diagnosis.py         Root-cause analysis combining anomaly, degradation, stress
    ├── rul_regressor.py     ForecastNet ensemble wrapper
    └── what_if.py           RUL projection for routes and multi-leg journeys
```

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Deep learning | PyTorch (ForecastNet CNN+Attention ensemble) |
| ML / features | scikit-learn, XGBoost, pandas, NumPy |
| Agent tooling | LangChain (tool dispatch), FastMCP |
| API | FastAPI, Uvicorn, Pydantic |
| Database | SQLite (via stdlib `sqlite3`) |
| Environment | python-dotenv |
| Lint / test | Ruff, pytest |

---

## License

[MIT](LICENSE) © James Scott
