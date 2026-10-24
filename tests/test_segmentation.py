import numpy as np
import pytest

from phenoscope.evaluation import average_precision, evaluate_instances, match_instances
from phenoscope.segmentation import (
    available_segmenters,
    clean_labels,
    get_segmenter,
    masks_to_labels,
)


def test_registry_lists_all_backends():
    assert {"watershed", "cellpose", "sam", "unet"} <= set(available_segmenters())
    with pytest.raises(KeyError):
        get_segmenter("nope")


def test_watershed_on_synthetic_nuclei(cells):
    img, gt = cells
    res = get_segmenter("watershed").run(img[:1])
    m = evaluate_instances(gt, res.labels)
    assert m.ap50 > 0.9
    assert abs(res.n_objects - m.n_gt) <= 2
    assert res.runtime_s > 0


def test_ap_perfect_and_empty():
    gt = np.zeros((20, 20), int)
    gt[2:8, 2:8] = 1
    gt[10:18, 10:18] = 2
    assert average_precision(gt, gt, (0.5, 0.9)) == [1.0, 1.0]
    assert match_instances(gt, np.zeros_like(gt)) == (0, 0, 2)


def test_ap_counts_fp_and_fn():
    gt = np.zeros((20, 20), int)
    gt[0:5, 0:5] = 1
    gt[10:15, 10:15] = 2
    pred = np.zeros_like(gt)
    pred[0:5, 0:5] = 7          # TP (non-contiguous id)
    pred[15:20, 0:5] = 3        # FP
    tp, fp, fn = match_instances(gt, pred, 0.5)
    assert (tp, fp, fn) == (1, 1, 1)
    assert average_precision(gt, pred, (0.5,))[0] == pytest.approx(1 / 3)


def test_clean_labels_removes_small_and_fills_holes():
    lab = np.zeros((30, 30), int)
    lab[5:15, 5:15] = 4
    lab[9:11, 9:11] = 0          # hole
    lab[25, 25] = 9              # tiny
    out = clean_labels(lab, min_size=5)
    assert out.max() == 1
    assert out[10, 10] == 1
    assert out[25, 25] == 0


def test_masks_to_labels_resolves_overlap_by_score():
    a = np.zeros((10, 10), bool)
    a[0:6, 0:6] = True
    b = np.zeros((10, 10), bool)
    b[3:9, 3:9] = True
    lab = masks_to_labels([a, b], scores=[0.2, 0.9])
    assert lab.max() == 2
    assert lab[4, 4] == lab[8, 8]  # high-score mask wins the overlap
