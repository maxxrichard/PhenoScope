from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from phenoscope.api.schemas import JobOut
from phenoscope.jobs.backends import get_backend
from phenoscope.jobs.tasks import TASKS

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobSubmit(BaseModel):
    kind: str
    payload: dict[str, Any]


@router.post("", response_model=JobOut, status_code=202)
def submit(body: JobSubmit) -> JobOut:
    if body.kind not in TASKS:
        raise HTTPException(400, f"unknown job kind {body.kind!r}; available: {sorted(TASKS)}")
    backend = get_backend()
    return JobOut(**backend.status(backend.submit(body.kind, body.payload)))


@router.get("/{job_id}", response_model=JobOut)
def status(job_id: str) -> JobOut:
    st = get_backend().status(job_id)
    if st is None:
        raise HTTPException(404, "job not found")
    return JobOut(**st)
