"""Integration tests — end-to-end pipeline from DB through feature builder."""

import pytest


def test_load_and_feature_pipeline():
    """End-to-end: DB → load_processed → build_features returns expected shape."""
    from src.features.loader import load_processed
    from src.features.builder import build_features

    df = load_processed("FD004", "test", add_op_settings=True)
    assert len(df) > 0, "load_processed returned empty DataFrame"
    assert "unit" in df.columns, "'unit' column missing"

    feat = build_features(df.head(500), add_rolling=False)
    assert len(feat) > 0, "build_features returned empty DataFrame"


def test_rul_labels_range():
    """RUL labels from DB should be in range 6–195 for FD004."""
    from src.db.database import get_rul_labels

    labels = get_rul_labels("FD004")
    assert len(labels) == 248, f"Expected 248 engines, got {len(labels)}"
    assert min(labels.values()) >= 6, f"Min RUL too low: {min(labels.values())}"
    assert max(labels.values()) <= 195, f"Max RUL too high: {max(labels.values())}"


def test_db_async_jobs_crud():
    """create_job / update_job / get_job round-trip works correctly."""
    import uuid
    from src.db.database import create_job, update_job, get_job

    job_id = str(uuid.uuid4())
    create_job(job_id, engine_id=42)

    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "queued"
    assert job["engine_id"] == 42

    update_job(job_id, "done", {"recommendation": "CLEARED"})
    job = get_job(job_id)
    assert job["status"] == "done"
    assert job["result"]["recommendation"] == "CLEARED"


def test_get_job_missing_returns_none():
    """get_job with unknown ID returns None (no exception)."""
    from src.db.database import get_job

    result = get_job("nonexistent-job-id-xyz")
    assert result is None
