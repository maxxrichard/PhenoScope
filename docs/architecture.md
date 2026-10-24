# Architecture

## Design goals

1. **One pixel format.** Everything downstream of ingestion reads OME-Zarr (NGFF 0.4). Images are chunked multiscale pyramids; masks live next to them under `labels/<name>`. The same code path works on local disk and S3 (`fsspec`), so a notebook, the API and a GPU pod all read data the same way.
2. **Swappable models.** Segmenters register by name behind one interface (`Segmenter.run(image_cyx) -> SegmentationResult`). Benchmarks, the API, the napari widget and Airflow only refer to a name and a parameter dict, so they don't need to know which backend runs.
3. **Benchmark before release.** Every model change is scored on curated benchmark sets (AP@0.5, mAP@0.5:0.95). Every new imaging batch is checked for drift against the reference distribution of those sets before results go to scientists.
4. **Cheap paths stay cheap.** Heavy dependencies (torch, Cellpose, SAM, MONAI, timm, OpenSlide, napari, Airflow) are optional extras and only imported when used. The core package and the full test suite run on a laptop CPU.

## Components

### Ingestion and storage — `phenoscope.io`
- `write_ome_zarr` builds a pyramid (2× mean downsampling, Blosc-zstd), writes `multiscales` + `omero` channel metadata and optional provenance under `phenoscope`.
- `write_labels` attaches instance masks with nearest-neighbour pyramids and `image-label` metadata, so napari-ome-zarr, vizarr and QuPath display them as label layers.

### Preprocessing — `phenoscope.preprocessing`
- Percentile normalisation and robust z-scores.
- Retrospective flat-field estimation per plate and channel (median over fields + Gaussian smoothing).
- Macenko stain normalisation and H/E separation for whole-slide tiles.

### Segmentation — `phenoscope.segmentation`
| Backend | Use | Notes |
|---|---|---|
| `watershed` | baseline, CI | Otsu + distance-transform seeds |
| `cellpose` | generalist cells and nuclei | `cyto3` / `nuclei` or custom weights |
| `sam` | dense nuclei, H&E | automatic mask generator + fine-tuned mask decoder, overlap resolution by score |
| `unet` | tissue nuclei | MONAI U-Net, background/interior/boundary head → seeded watershed |

`segmentation.training` has the U-Net training loop (DiceCE, dihedral augmentation, cosine LR, best-AP checkpointing) and SAM mask-decoder fine-tuning with box prompts from ground-truth instances. The image encoder stays frozen and embeddings are cached.

### Profiling — `phenoscope.profiling`
- `extract_features`: morphology, per-channel intensity (mean/min/max/std/integrated), and optional GLCM texture.
- `ViTEmbedder`: a `timm` ViT (DINO by default). Multichannel fluorescence is embedded one channel at a time and concatenated. H&E is embedded as RGB.
- `aggregate_profiles` / `normalize_to_controls` / `feature_select`: well-level median profiles, robust z-scores against negative controls, and correlation pruning.

### Registration — `phenoscope.registration`
Cyclic multiplex rounds are aligned to cycle 0 on the nuclear reference channel. The steps are phase cross-correlation on a downsampled image, then sub-pixel phase correlation at full resolution, then optional ECC affine refinement. The transform is applied to every marker in that cycle, and NCC before and after registration is recorded for QC.

### Fusion — `phenoscope.fusion`
Sample-level image embeddings and log-expression are standardised and reduced with PCA per modality. Image-only, expression-only, early (concatenation) and late (probability averaging) models are compared with out-of-fold AUROC. When donors or plates are passed as `groups`, the folds are grouped by them.

### Monitoring — `phenoscope.monitoring`
`DriftMonitor` compares a batch to a frozen reference:
- **QC features** (channel percentiles, dynamic range, saturation): a feature alerts only when PSI ≥ 0.25 *and* KS p < 0.01. A batch is flagged when at least 2 features alert.
- **Embeddings**: unbiased RBF-MMD² with a median-heuristic bandwidth and a permutation test.

Staining-batch shifts show up first as intensity-percentile drift. Morphology drift without intensity drift usually means a real biological change.

### Serving — `phenoscope.api`, `phenoscope.web`, `phenoscope.viewer`
FastAPI handles datasets and images, synchronous and asynchronous segmentation, features, PNG rendering for the web viewer, jobs and drift checks. The job backend is either a local thread pool or Kubernetes `batch/v1` Jobs with GPU requests and tolerations. The napari plugin runs the same segmenters locally or browses a remote server.

### Orchestration — `pipelines/airflow`
`phenoscope_profiling` runs discover → convert → drift gate (short-circuit) → batch → segment (GPU pods, mapped) → features + embeddings (GPU pods, mapped) → aggregate.
