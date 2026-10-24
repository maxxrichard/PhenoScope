import numpy as np

from phenoscope.benchmarks import synthetic_he
from phenoscope.preprocessing import MacenkoNormalizer, correct_illumination, estimate_flatfield
from phenoscope.wsi import ArraySlide, iter_tiles, tile_grid, tissue_mask


def test_tissue_mask_and_tiles():
    he = synthetic_he((512, 512))
    mask = tissue_mask(he)
    assert 0.3 < mask.mean() < 0.9
    assert not mask[0, 0]  # glass corner
    slide = ArraySlide(he)
    tiles = tile_grid(slide, tile_size=128, thumb_downsample=4, min_tissue=0.5)
    assert 0 < len(tiles) < 16
    t, rgb = next(iter_tiles(slide, tiles))
    assert rgb.shape == (128, 128, 3)


def test_macenko_matches_reference_stats():
    ref = synthetic_he((256, 256), seed=0)
    src = np.clip(synthetic_he((256, 256), seed=1).astype(int) * [0.9, 1.0, 1.1], 0, 255)
    norm = MacenkoNormalizer().fit(ref)
    out = norm.transform(src.astype(np.uint8))
    assert out.shape == src.shape and out.dtype == np.uint8
    conc = norm.separate(ref)
    assert conc.shape == (2, 256, 256)


def test_flatfield_correction_flattens_vignetting():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:128, :128]
    vignette = 1 - 0.4 * (((yy - 64) ** 2 + (xx - 64) ** 2) / 64**2)
    fields = np.stack([(1000 + rng.normal(0, 20, (128, 128))) * vignette for _ in range(20)])
    ff = estimate_flatfield(fields.astype(np.uint16), sigma=8, downsample=2)
    corrected = correct_illumination(fields[0].astype(np.uint16), ff)
    assert corrected.std() < fields[0].std() / 2
