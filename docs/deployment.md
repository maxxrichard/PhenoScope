# Deployment

## Local

```bash
cp .env.example .env
docker compose up --build
# API + viewer: http://localhost:8000   MinIO console: http://localhost:9001
```

## AWS EKS

1. **Cluster:** `eksctl create cluster -f deploy/eks/cluster.yaml`. This creates a CPU node group (`phenoscope/pool=cpu`) and a GPU node group that scales from zero (`phenoscope/pool=gpu`, tainted `nvidia.com/gpu`). Install the NVIDIA device plugin and cluster-autoscaler.
2. **Buckets:** `phenoscope-raw`, `phenoscope-zarr`, `phenoscope-benchmarks`, `phenoscope-models`. Bind the IAM roles for the `phenoscope-api` and `phenoscope-worker` service accounts with IRSA (`deploy/k8s/overlays/eks`).
3. **Secrets:** create `Secret/phenoscope-env` with `PHENOSCOPE_DATABASE_URL` (RDS PostgreSQL) and `PHENOSCOPE_MONGO_URL`, for example through External Secrets.
4. **Images:**
   ```bash
   docker build -f deploy/docker/Dockerfile -t ghcr.io/maxxrichard/phenoscope:0.4.0 .
   docker build -f deploy/docker/Dockerfile.gpu -t ghcr.io/maxxrichard/phenoscope-gpu:0.4.0 .
   ```
5. **Apply:** `kubectl apply -k deploy/k8s/overlays/eks`
6. **Airflow:** deploy the DAGs in `pipelines/airflow/dags` (for example with the official Helm chart and git-sync). They use `KubernetesPodOperator` in the `phenoscope` namespace.

GPU worker pods pull model weights from `s3://phenoscope-models/current/` in an init container, so a model rollout only needs an S3 sync and no image rebuild.
