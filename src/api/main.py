"""Engine Health FastAPI — HTTP interface for the agent and direct prediction endpoints.

Endpoints:
    POST /agent/query         — async agent query; returns job_id immediately
    GET  /agent/result/{id}   — poll for agent result
    POST /predict/rul         — synchronous single-engine RUL prediction
    GET  /engine/{id}/history — prediction history from agent_runs.jsonl
    GET  /health              — liveness check

Run with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

_ROOT    = Path(__file__).resolve().parents[2]
_RUNS_FP = _ROOT / "outputs" / "agent_runs.jsonl"

app = FastAPI(
    title="Engine Health API",
    description="Predictive maintenance agent for NASA CMAPSS FD004 turbofan engines.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# In-memory job store (no DB needed in this phase)
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------

_reg = None

def _get_reg():
    global _reg
    if _reg is None:
        from src.models.rul_regressor import RULRegressor
        _reg = RULRegressor.load()
    return _reg


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class AgentQueryRequest(BaseModel):
    engine_id: int = Field(..., description="Engine unit number (1–248 test, 1–249 train).")
    query_text: str = Field(..., description="Natural language question about the engine.")


class AgentQueryResponse(BaseModel):
    job_id: str
    status: str = "queued"


class AgentResultResponse(BaseModel):
    job_id: str
    status: str
    engine_id: Optional[int] = None
    query: Optional[str] = None
    recommendation: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    turns: Optional[int] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class RULRequest(BaseModel):
    engine_id: int = Field(..., description="Engine unit number.")


class RULResponse(BaseModel):
    engine_id: int
    rul: float
    lower_80: float
    upper_80: float
    std: float
    current_cycle: int
    rul_true: Optional[float] = None


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

def _run_agent(job_id: str, engine_id: int, query_text: str) -> None:
    """Synchronous wrapper called in a thread pool executor."""
    _jobs[job_id]["status"] = "running"
    try:
        from src.agent.orchestrator import AgentOrchestrator
        agent = AgentOrchestrator()
        result = agent.run(engine_id=engine_id, query_text=query_text)
        _jobs[job_id].update({
            "status": "done" if result.get("error") is None else "error",
            **result,
        })
    except Exception as exc:
        _jobs[job_id].update({"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/agent/query", response_model=AgentQueryResponse, status_code=202)
async def agent_query(req: AgentQueryRequest):
    """Submit an engine health query to the LLM agent.

    Returns a ``job_id`` immediately.  Poll ``GET /agent/result/{job_id}`` for the result.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "engine_id": req.engine_id,
        "query": req.query_text,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_agent, job_id, req.engine_id, req.query_text)
    return AgentQueryResponse(job_id=job_id)


@app.get("/agent/result/{job_id}", response_model=AgentResultResponse)
def agent_result(job_id: str):
    """Poll for the result of a previously submitted agent query."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    job = _jobs[job_id]
    return AgentResultResponse(
        job_id=job_id,
        status=job["status"],
        engine_id=job.get("engine_id"),
        query=job.get("query"),
        recommendation=job.get("recommendation"),
        tool_calls=job.get("tool_calls"),
        turns=job.get("turns"),
        latency_ms=job.get("latency_ms"),
        error=job.get("error"),
    )


@app.post("/predict/rul", response_model=RULResponse)
def predict_rul(req: RULRequest):
    """Synchronous RUL prediction for a single engine (no agent reasoning)."""
    try:
        from src.mcp_server.server import estimate_rul
        result = estimate_rul(req.engine_id)
        return RULResponse(engine_id=req.engine_id, **result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/engine/{engine_id}/history")
def engine_history(engine_id: int):
    """Return all agent-run records for the given engine (from agent_runs.jsonl)."""
    if not _RUNS_FP.exists():
        return {"engine_id": engine_id, "runs": []}
    runs = []
    with open(_RUNS_FP, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("engine_id") == engine_id:
                    runs.append(record)
            except json.JSONDecodeError:
                continue
    return {"engine_id": engine_id, "runs": runs, "count": len(runs)}
