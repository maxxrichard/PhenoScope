"""Nightly image-to-feature pipeline for newly acquired plates / slides.

ingest → convert to OME-Zarr (S3) → QC + drift check → segment (GPU pod)
       → features + ViT embeddings (GPU pod) → aggregate profiles → load metadata DB

Segmentation and embedding run as Kubernetes pods on the EKS GPU node group via
``KubernetesPodOperator`` with dynamic task mapping (one pod per image batch).
A drift alert short-circuits the model steps so a staining-batch shift is reviewed
before results are released.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

IMAGE = "ghcr.io/maxxrichard/phenoscope-gpu:{{ var.value.get('phenoscope_version', 'latest') }}"
NAMESPACE = "phenoscope"
GPU = k8s.V1ResourceRequirements(
    requests={"cpu": "4", "memory": "16Gi"}, limits={"memory": "32Gi", "nvidia.com/gpu": "1"}
)
GPU_TOLERATION = [k8s.V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")]
ENV_FROM = [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="phenoscope-env"))]


@dag(
    dag_id="phenoscope_profiling",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    params={
        "inbox": Param("s3://phenoscope-raw/inbox/", type="string"),
        "dest": Param("s3://phenoscope-zarr/", type="string"),
        "benchmark": Param("hcs-cellpainting", type="string"),
        "segmenter": Param("cellpose", enum=["cellpose", "sam", "unet", "watershed"]),
        "batch_size": Param(64, type="integer"),
    },
    tags=["phenoscope", "imaging"],
)
def phenoscope_profiling():
    @task
    def discover(params=None) -> list[str]:
        import fsspec

        fs, root = fsspec.core.url_to_fs(params["inbox"])
        return [f"s3://{p}" for p in fs.glob(f"{root}/**/*.tif*")]

    @task
    def convert(paths: list[str], params=None) -> list[str]:
        import tempfile
        from pathlib import Path

        import fsspec

        from phenoscope.io import read_image, write_ome_zarr

        out = []
        for p in paths:
            with fsspec.open(p, "rb") as fh, tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
                tmp.write(fh.read())
                tmp.flush()
                data = read_image(tmp.name)
            uri = f"{params['dest'].rstrip('/')}/{Path(p).stem}.zarr"
            write_ome_zarr(data, uri, name=Path(p).stem)
            out.append(uri)
        return out

    @task.short_circuit
    def drift_gate(uris: list[str], params=None) -> bool:
        import pandas as pd

        from phenoscope.io import read_ome_zarr
        from phenoscope.monitoring import DriftMonitor
        from phenoscope.monitoring.drift import image_qc_features

        rows = []
        for u in uris:
            img = read_ome_zarr(u, level=2)
            rows.append(image_qc_features(img.data, img.channel_names))
        ref = pd.read_parquet(f"s3://phenoscope-benchmarks/{params['benchmark']}/qc_reference.parquet")
        run_id = get_current_context()["run_id"]
        report = DriftMonitor(ref).check(pd.DataFrame(rows), batch_id=run_id)
        print(json.dumps(report.as_dict(), default=str)[:5000])
        return not report.drifted

    @task
    def batches(uris: list[str], params=None) -> list[str]:
        n = params["batch_size"]
        return [json.dumps(uris[i : i + n]) for i in range(0, len(uris), n)]

    def gpu_pod(task_id: str, kind: str, extra: dict) -> KubernetesPodOperator:
        return KubernetesPodOperator.partial(
            task_id=task_id,
            namespace=NAMESPACE,
            image=IMAGE,
            cmds=["python", "-m", "phenoscope.pipelines.batch"],
            container_resources=GPU,
            tolerations=GPU_TOLERATION,
            node_selector={"phenoscope/pool": "gpu"},
            env_from=ENV_FROM,
            get_logs=True,
            on_finish_action="delete_succeeded_pod",
            env_vars={"PHENOSCOPE_TASK": kind, "PHENOSCOPE_TASK_PARAMS": json.dumps(extra)},
        )

    @task
    def aggregate(uris: list[str]) -> str:
        from phenoscope.pipelines.batch import aggregate_run

        return aggregate_run(uris)

    uris = convert(discover())
    ok = drift_gate(uris)
    b = batches(uris)
    ok >> b
    seg = gpu_pod("segment", "segment", {"segmenter": "{{ params.segmenter }}"}).expand(
        arguments=b.map(lambda x: [x])
    )
    feat = gpu_pod("features_embed", "features+embed", {}).expand(arguments=b.map(lambda x: [x]))
    seg >> feat >> aggregate(uris)


phenoscope_profiling()
