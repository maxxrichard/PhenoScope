import numpy as np

from phenoscope.benchmarks import synthetic_multiplex
from phenoscope.registration import register_cycles


def test_register_twelve_markers_recovers_shifts():
    cycles, shifts = synthetic_multiplex(n_cycles=4, markers_per_cycle=3, seed=3)
    res = register_cycles(cycles, affine=False)
    assert res.image.shape[0] == 13  # DAPI + 12 markers
    assert res.channel_names[0] == "DAPI"
    for warp, (dy, dx) in zip(res.transforms, shifts):
        assert abs(warp[0, 2] - dx) < 0.5
        assert abs(warp[1, 2] - dy) < 0.5
    s = res.summary()
    assert s["mean_ncc_after"] > 0.95 > s["mean_ncc_before"]


def test_affine_refinement_keeps_alignment():
    cycles, _ = synthetic_multiplex(n_cycles=2, markers_per_cycle=1, seed=5)
    res = register_cycles(cycles, affine=True, keep_reference="all")
    assert res.channel_names == ["DAPI_c0", "M00", "DAPI_c1", "M01"]
    assert np.min(res.ncc_after) > 0.95
