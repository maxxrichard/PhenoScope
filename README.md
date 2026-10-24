# PhenoScope

**Imaging AI platform for microscopy and digital pathology.**

PhenoScope turns large-scale scientific imaging into something people can actually use: scalable pipelines and web tools for **segmentation, phenotypic profiling and registration** of fluorescence / brightfield microscopy, high-content screens and H&E whole-slide images, fused with genomic and clinical data, with benchmark-driven evaluation and drift monitoring.

[![CI](https://github.com/maxxrichard/PhenoScope/actions/workflows/ci.yml/badge.svg)](https://github.com/maxxrichard/PhenoScope/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`Python` · `PyTorch` · `MONAI` · `Cellpose` · `SAM` · `OpenCV` · `scikit-image` · `OME-Zarr` · `Napari` · `FastAPI` · `Kubernetes` · `AWS S3`

---

## Highlights

| Area | What it does |
|---|---|
| **Cell & tissue segmentation** | One interface over Cellpose, fine-tuned SAM (mask-decoder fine-tuning), a MONAI 3-class U-Net and a classical watershed baseline. |
| **Phenotypic profiling** | Interpretable per-object features (morphology, intensity, GLCM texture) plus ViT (DINO) embeddings of single-cell crops and WSI tiles; well-level aggregation and control-based normalisation. |
| **Multiplex registration** | Cyclic (CyCIF/CODEX-style) registration on the nuclear reference channel: phase correlation → sub-pixel refinement → ECC affine; stacks e.g. 12 markers into one image. |
| **Digital pathology** | Tissue detection, tiling and Macenko stain normalisation for H&E whole-slide images (OpenSlide). |
| **Multimodal fusion** | Early/late fusion of image embeddings with transcriptomics, evaluated with grouped cross-validated AUROC. |
| **Benchmarks & drift** | Curated benchmark registry with AP@0.5 / mAP evaluation; per-batch drift checks (PSI + KS on QC features, kernel MMD on embeddings) that catch staining-batch shifts before release. |
| **Data layer** | OME-Zarr (NGFF 0.4) on S3, PostgreSQL + MongoDB metadata, Airflow DAGs, GPU jobs on Kubernetes (AWS EKS). |
| **Tools for scientists** | FastAPI backend, a no-build web viewer, and a napari plugin. |

### Results (internal benchmarks)

| Metric | Value |
|---|---|
| Instance-segmentation AP (fine-tuned SAM / U-Net) | **0.78**, +9 pp over the Cellpose baseline |
| Images profiled | ~120k fluorescence / brightfield images + 800 H&E WSIs |
| Multiplexed registration | 12-marker cyclic imaging |
| Data under management | 35 TB as OME-Zarr on S3 |
| Phenotype-classification AUROC from image + transcriptomics fusion | **+0.06** over image-only |
| Drift monitoring | caught 2 staining-batch shifts before release |
| Image-to-feature turnaround | **2 days → 3 hours** |

These numbers come from the project's private datasets, which aren't part of this repository. The test suite and `phenoscope benchmark synthetic` run on synthetic data with exact ground truth.

---

## Architecture

```text
            acquisition (TIFF / OME-TIFF / SVS / NDPI)
                              │
                 Airflow DAG: phenoscope_profiling
                              │
     ┌────────── convert ─────┴───── QC + drift gate ──────────┐
     │   OME-Zarr pyramids on S3                PSI/KS + MMD    │
     ▼                                                          ▼
  segment (GPU pod, EKS)  ──►  features + ViT embeddings  ──►  aggregate profiles
  Cellpose │ SAM-ft │ U-Net        (GPU pod, EKS)               (per well / sample)
     │                                  │                          │
     └──── labels/ in OME-Zarr ─────────┴─── parquet / npz ────────┤
                                                                   ▼
             PostgreSQL (datasets, images, runs, drift)    fusion with transcriptomics
             MongoDB (plate maps, acquisition metadata)
                              │
                     FastAPI  ─┬─ web viewer
                               └─ napari plugin
```

More detail: [docs/architecture.md](docs/architecture.md) · [docs/data-model.md](docs/data-model.md) · [docs/benchmarks.md](docs/benchmarks.md) · [docs/deployment.md](docs/deployment.md)

## Repository layout

```text
src/phenoscope/
  io/             OME-Zarr reader/writer (NGFF 0.4, labels), S3 storage, TIFF readers
  preprocessing/  percentile normalisation, flat-field correction, Macenko stain norm
  segmentation/   Segmenter registry: watershed, cellpose, sam, unet; training loops
  evaluation/     instance AP (Hungarian matching), precision/recall/F1
  profiling/      hand-crafted features, ViT embeddings, aggregation, feature selection
  registration/   cyclic multiplex registration
  wsi/            slide readers, tissue mask, tiling
  fusion/         image × transcriptomics fusion with grouped CV
  monitoring/     drift detection (PSI, KS, MMD)
  benchmarks/     dataset registry, runner, synthetic data generators
  db/             SQLAlchemy models (PostgreSQL/SQLite), MongoDB metadata store
  jobs/           local thread-pool and Kubernetes job backends
  api/            FastAPI app and routes
  web/            static web viewer
  viewer/         napari plugin (manifest + widgets)
  pipelines/      batch worker used by Airflow pods
pipelines/airflow/dags/   Airflow DAGs
deploy/                   Dockerfiles, Kubernetes (kustomize) manifests, eksctl cluster
configs/                  model and benchmark configs
tests/                    pytest suite (CPU only, synthetic data)
```

## Quickstart

```bash
git clone https://github.com/maxxrichard/PhenoScope.git && cd PhenoScope
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + tests (CPU only)
pip install -e ".[ml]"             # Cellpose, SAM, MONAI, timm (optional, GPU recommended)
```

Create a demo dataset and open the web viewer:

```bash
phenoscope demo
phenoscope serve   # http://localhost:8000  (API docs at /docs)
```

### CLI

```bash
# Convert an image to multiscale OME-Zarr (local or s3://)
phenoscope convert plate1_A01.tif data/plate1_A01.zarr --channels DAPI,CellMask --pixel-size 0.65

# Segment (writes labels/cells into the store)
phenoscope segment data/plate1_A01.zarr --segmenter cellpose --channels DAPI,CellMask \
    --params '{"diameter": 30}'

# Per-object features
phenoscope features data/plate1_A01.zarr data/plate1_A01_features.parquet --texture

# Register 4 cycles × (DAPI + 3 markers) into one 12-marker image
phenoscope register cycle0.zarr cycle1.zarr cycle2.zarr cycle3.zarr --out registered.zarr

# Compare segmenters on a curated benchmark
phenoscope benchmark hcs-cellpainting --models configs/models.yaml --baseline cellpose-cyto3

# Drift check of a new batch against a benchmark reference (exit code 2 on drift)
phenoscope drift reference_qc.csv batch_qc.csv --batch-id stain-batch-07
```

### Python API

```python
from phenoscope.io import read_ome_zarr
from phenoscope.segmentation import get_segmenter
from phenoscope.evaluation import evaluate_instances
from phenoscope.profiling import extract_features, ViTEmbedder, extract_object_crops

img = read_ome_zarr("data/plate1_A01.zarr")
labels = get_segmenter("sam", decoder_weights="models/sam_decoder_ft.pt").segment(img.data)

features = extract_features(img.data, labels, img.channel_names, pixel_size_um=0.65)
crops, ids = extract_object_crops(img.data, labels, size=96)
embeddings = ViTEmbedder().embed(crops)          # (n_cells, n_channels * 384)
```

### REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health`, `/segmenters` | status, available backends |
| `POST/GET` | `/datasets` | create / list datasets |
| `POST/GET` | `/datasets/{id}/images` | register OME-Zarr images / list them (filter by plate, batch) |
| `POST` | `/images/{id}/segment` | run segmentation (sync, or `asynchronous: true` → job) |
| `GET` | `/images/{id}/render.png` | composite preview with label outlines |
| `GET` | `/images/{id}/features` | per-object features |
| `POST/GET` | `/jobs`, `/jobs/{id}` | submit / poll background jobs (local or Kubernetes) |
| `POST` | `/drift/check`, `GET /drift/history` | batch drift checks |

### napari

After `pip install -e ".[viewer]"`, the **PhenoScope** plugin adds *Segment cells*, *Measure features* and *PhenoScope server* widgets (browse datasets on a running API and open OME-Zarr images).

## Local stack and deployment

```bash
docker compose up --build      # API + PostgreSQL + MongoDB + MinIO (S3)
```

Production runs on AWS EKS: a CPU node group for the API and Airflow, and an autoscaled GPU node group (`g5`) for segmentation and embedding pods. See [docs/deployment.md](docs/deployment.md).

## Development

```bash
make install   # editable install with dev extras
make test      # pytest
make lint      # ruff
```

## License

MIT, see [LICENSE](LICENSE).
