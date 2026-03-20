"""Engine Health FastAPI — HTTP interface for the agent and direct prediction endpoints.

Endpoints:
    POST /agent/query         — async agent query; returns job_id immediately
    GET  /agent/result/{id}   — poll for agent result
    POST /predict/rul         — synchronous single-engine RUL prediction
    GET  /engine/{id}/history — prediction history from agent_runs DB table
    GET  /health              — liveness check

Run with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.logging_config import configure_logging

load_dotenv()
configure_logging()
warnings.filterwarnings("ignore")

app = FastAPI(
    title="Engine Health API",
    description="Predictive maintenance agent for NASA CMAPSS FD004 turbofan engines.",
    version="1.0.0",
)


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
    from src.db.database import update_job
    update_job(job_id, "running")
    try:
        from src.agent.orchestrator import AgentOrchestrator
        agent = AgentOrchestrator()
        result = agent.run(engine_id=engine_id, query_text=query_text)
        status = "done" if result.get("error") is None else "error"
        update_job(job_id, status, result)
    except Exception as exc:
        update_job(job_id, "error", {"error": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/agent/query", response_model=AgentQueryResponse, status_code=202)
async def agent_query(req: AgentQueryRequest):
    """Submit an engine health query to the agent.

    Returns a ``job_id`` immediately.  Poll ``GET /agent/result/{job_id}`` for the result.
    """
    from src.db.database import create_job
    job_id = str(uuid.uuid4())
    create_job(job_id, req.engine_id)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_agent, job_id, req.engine_id, req.query_text)
    return AgentQueryResponse(job_id=job_id)


@app.get("/agent/result/{job_id}", response_model=AgentResultResponse)
def agent_result(job_id: str):
    """Poll for the result of a previously submitted agent query."""
    from src.db.database import get_job
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    result = job.get("result") or {}
    return AgentResultResponse(
        job_id=job_id,
        status=job["status"],
        engine_id=job.get("engine_id"),
        query=result.get("query"),
        recommendation=result.get("recommendation"),
        tool_calls=result.get("tool_calls"),
        turns=result.get("turns"),
        latency_ms=result.get("latency_ms"),
        error=result.get("error"),
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
    """Return all agent-run records for the given engine from the DB."""
    from src.db.database import get_db
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, query, recommendation, turns, latency_ms, error "
        "FROM agent_runs WHERE engine_id=? ORDER BY id DESC LIMIT 100",
        (engine_id,),
    ).fetchall()
    conn.close()
    return {
        "engine_id": engine_id,
        "runs": [dict(r) for r in rows],
        "count": len(rows),
    }
