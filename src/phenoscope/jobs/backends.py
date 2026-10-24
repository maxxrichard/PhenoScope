"""Job execution backends.

* :class:`LocalBackend` – thread pool inside the API process (dev, small images).
* :class:`KubernetesBackend` – one ``batch/v1`` Job per task on the GPU node group
  (AWS EKS in production); the pod runs ``phenoscope run-task <kind> <payload>``.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from phenoscope.config import Settings, get_settings
from phenoscope.db.models import Job
from phenoscope.db.session import get_session
from phenoscope.jobs.tasks import run_task

log = logging.getLogger(__name__)


class JobBackend(ABC):
    name: str

    def submit(self, kind: str, payload: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:16]
        with get_session() as s:
            s.add(Job(id=job_id, kind=kind, backend=self.name, payload=payload, status="queued"))
        self._launch(job_id, kind, payload)
        return job_id

    @abstractmethod
    def _launch(self, job_id: str, kind: str, payload: dict[str, Any]) -> None: ...

    def status(self, job_id: str) -> dict[str, Any] | None:
        with get_session() as s:
            job = s.get(Job, job_id)
            if job is None:
                return None
            return {"id": job.id, "kind": job.kind, "status": job.status, "backend": job.backend,
                    "result": job.result, "error": job.error,
                    "created_at": job.created_at.isoformat()}


def _update(job_id: str, **fields: Any) -> None:
    with get_session() as s:
        job = s.get(Job, job_id)
        for k, v in fields.items():
            setattr(job, k, v)


class LocalBackend(JobBackend):
    name = "local"

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="phenoscope-job")

    def _launch(self, job_id: str, kind: str, payload: dict[str, Any]) -> None:
        self._pool.submit(self._run, job_id, kind, payload)

    @staticmethod
    def _run(job_id: str, kind: str, payload: dict[str, Any]) -> None:
        _update(job_id, status="running")
        try:
            result = run_task(kind, payload)
            _update(job_id, status="succeeded", result=result)
        except Exception as exc:  # noqa: BLE001 - surfaced through the job record
            log.exception("job %s failed", job_id)
            _update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class KubernetesBackend(JobBackend):
    name = "kubernetes"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def manifest(self, job_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        gpu = kind in {"segment", "embed"}
        resources = {"requests": {"cpu": "4", "memory": "16Gi"}, "limits": {"memory": "32Gi"}}
        if gpu:
            resources["limits"]["nvidia.com/gpu"] = "1"
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"phenoscope-{kind}-{job_id}",
                "namespace": self.settings.k8s_namespace,
                "labels": {"app.kubernetes.io/part-of": "phenoscope", "phenoscope/job-id": job_id},
            },
            "spec": {
                "backoffLimit": 2,
                "ttlSecondsAfterFinished": 3600,
                "template": {
                    "metadata": {"labels": {"phenoscope/job-id": job_id}},
                    "spec": {
                        "restartPolicy": "Never",
                        "serviceAccountName": "phenoscope-worker",
                        "nodeSelector": {"phenoscope/pool": "gpu" if gpu else "cpu"},
                        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists",
                                         "effect": "NoSchedule"}] if gpu else [],
                        "containers": [{
                            "name": "worker",
                            "image": self.settings.k8s_gpu_image,
                            "args": ["run-task", kind, json.dumps(payload), "--job-id", job_id],
                            "envFrom": [{"secretRef": {"name": "phenoscope-env"}}],
                            "resources": resources,
                        }],
                    },
                },
            },
        }

    def _launch(self, job_id: str, kind: str, payload: dict[str, Any]) -> None:
        try:
            from kubernetes import client, config
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Kubernetes backend needs `pip install phenoscope[cloud]`") from exc
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        client.BatchV1Api().create_namespaced_job(
            self.settings.k8s_namespace, self.manifest(job_id, kind, payload)
        )
        _update(job_id, status="submitted")


_BACKEND: JobBackend | None = None


def get_backend() -> JobBackend:
    global _BACKEND
    if _BACKEND is None:
        s = get_settings()
        _BACKEND = KubernetesBackend(s) if s.job_backend == "kubernetes" else LocalBackend(
            s.max_local_workers
        )
    return _BACKEND


def mark_job(job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
    """Called from inside a worker pod to report completion back to the metadata DB."""
    _update(job_id, status=status, result=result or {}, error=error)
