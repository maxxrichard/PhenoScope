# Changelog

## 0.4.0
- Fine-tuned SAM mask decoder and MONAI 3-class U-Net backends; benchmark runner reports the AP delta to a baseline.
- Image × transcriptomics fusion module with grouped cross-validated AUROC.
- Drift monitor: KS significance is now required alongside PSI, and PSI bins adapt to sample size.
- Airflow drift gate short-circuits model tasks on drifted batches.

## 0.3.0
- Cyclic multiplex registration (phase correlation + ECC affine), 12-marker stacks.
- H&E tiling, tissue detection, Macenko normalisation.

## 0.2.0
- FastAPI backend, web viewer, napari plugin; Kubernetes job backend.

## 0.1.0
- OME-Zarr I/O, Cellpose and watershed segmentation, hand-crafted features.
