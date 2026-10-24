# Benchmarks and evaluation

## Metric

Instance segmentation is scored with **AP at IoU threshold *t***:

```
AP@t = TP / (TP + FP + FN)
```

Predictions are matched one-to-one to ground truth with the Hungarian algorithm on the IoU matrix, and a match counts only if IoU ≥ *t*. This is the definition used by Cellpose, StarDist and the 2018 Data Science Bowl. The headline number is AP@0.5. The detailed number is mAP over IoU 0.50:0.05:0.95. Precision, recall and F1 at 0.5 are also reported (`phenoscope.evaluation.evaluate_instances`).

## Curated sets

`configs/benchmarks.yaml` registers three frozen benchmark sets:

| Name | Modality | Ground truth |
|---|---|---|
| `hcs-cellpainting` | 5-channel Cell Painting fluorescence | cell instances |
| `brightfield-cells` | label-free brightfield | cell instances |
| `he-nuclei` | H&E WSI tiles | nucleus instances |

Each set also stores `qc_reference.parquet`, the per-image QC-feature distribution the drift monitor compares new batches against.

## Running

```bash
phenoscope benchmark hcs-cellpainting --models configs/models.yaml --baseline cellpose-cyto3 --out results.csv
```

The output table includes `ap50_delta_pp`, the AP@0.5 difference to the baseline in percentage points.

## Fine-tuning

```python
from phenoscope.benchmarks import InstanceDataset, load_registry
from phenoscope.segmentation.training import TrainConfig, finetune_sam_decoder, train_unet

spec = load_registry()["he-nuclei"]
train, val = list(InstanceDataset(spec, "train")), list(InstanceDataset(spec, "val"))

train_unet(train, val, TrainConfig(epochs=150, out="models/unet_nuclei.pt"))
finetune_sam_decoder(train, checkpoint="models/sam_vit_b_01ec64.pth", out="models/sam_decoder_ft.pt")
```

## Drift monitoring

Before results from a new batch are released, the Airflow drift gate:
1. computes cheap QC features (channel percentiles, dynamic range, saturation) at a low pyramid level,
2. compares them to the benchmark reference using PSI (≥ 0.25) together with KS (p < 0.01),
3. optionally compares ViT embeddings with an MMD permutation test,
4. short-circuits the downstream model tasks if the batch drifted, so an analyst reviews it first.
