from __future__ import annotations

import pandas as pd
from fastapi import APIRouter
from sqlalchemy import select

from phenoscope.api.schemas import DriftOut, DriftRequest, FeatureDriftOut
from phenoscope.db.models import DriftCheck
from phenoscope.db.session import get_session
from phenoscope.monitoring.drift import DriftMonitor

router = APIRouter(prefix="/drift", tags=["monitoring"])


@router.post("/check", response_model=DriftOut)
def check(body: DriftRequest) -> DriftOut:
    monitor = DriftMonitor(pd.DataFrame(body.reference), features=body.features)
    report = monitor.check(pd.DataFrame(body.batch), batch_id=body.batch_id)
    with get_session() as s:
        s.add(DriftCheck(benchmark=body.benchmark, batch_id=body.batch_id, drifted=report.drifted,
                         mmd2=report.mmd2, mmd_p=report.mmd_p, report=report.as_dict()))
    return DriftOut(
        batch_id=report.batch_id,
        drifted=report.drifted,
        reasons=report.reasons,
        top_features=[FeatureDriftOut(feature=f.feature, psi=f.psi, ks_stat=f.ks_stat,
                                      ks_p=f.ks_p, level=f.level) for f in report.top(10)],
    )


@router.get("/history")
def history(benchmark: str | None = None, limit: int = 50) -> list[dict]:
    with get_session() as s:
        q = select(DriftCheck).order_by(DriftCheck.created_at.desc()).limit(limit)
        if benchmark:
            q = q.where(DriftCheck.benchmark == benchmark)
        return [{"id": c.id, "benchmark": c.benchmark, "batch_id": c.batch_id,
                 "drifted": c.drifted, "created_at": c.created_at.isoformat(),
                 "reasons": c.report.get("reasons", [])} for c in s.scalars(q)]
