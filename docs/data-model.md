# Data model

## Object storage (S3)

```text
s3://phenoscope-raw/inbox/...                 raw acquisitions (TIFF, OME-TIFF, SVS, NDPI)
s3://phenoscope-zarr/<plate>_<well>.zarr/     one OME-Zarr store per field / slide
    .zattrs                                   multiscales, omero channels, phenoscope provenance
    0/ 1/ 2/ 3/                               pyramid levels (c, y, x)
    labels/cells/0..3                         instance masks (y, x), uint32
    labels/nuclei/0..3
    profiles/features.parquet                 per-object hand-crafted features
    profiles/vit.npz                          per-object ViT embeddings + label ids
s3://phenoscope-zarr/_profiles/well_profiles.parquet
s3://phenoscope-benchmarks/<name>/            curated benchmark images + qc_reference.parquet
s3://phenoscope-models/current/               model weights pulled by worker pods
```

## Relational metadata (PostgreSQL)

| Table | Key columns |
|---|---|
| `datasets` | `name`, `modality` (fluorescence / brightfield / he / multiplex), `uri` |
| `images` | `dataset_id`, `uri`, `plate`, `well`, `field_of_view`, `sample_id`, `batch_id`, `width`, `height`, `channels`, `pixel_size_um` |
| `segmentation_runs` | `image_id`, `model`, `model_version`, `params`, `label_name`, `n_objects`, `runtime_s` |
| `feature_sets` | `segmentation_id`, `kind` (handcrafted / vit), `uri`, `n_objects`, `n_features` |
| `drift_checks` | `benchmark`, `batch_id`, `drifted`, `mmd2`, `mmd_p`, `report` (JSON) |
| `jobs` | `id`, `kind`, `status`, `backend`, `payload`, `result`, `error` |

Pixels are never stored in the database. It only holds URIs.

## Document metadata (MongoDB)

- `acquisitions`: one document per image URI with microscope settings, objective, exposure times, OMERO exports, and anything vendor-specific.
- `plate_maps`: `(plate, well)` → treatment, concentration, control flag, cell line, donor.
