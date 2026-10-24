import numpy as np
import zarr

from phenoscope.io import read_image, read_ome_zarr
from phenoscope.io.omezarr import list_levels, read_labels


def test_ome_zarr_roundtrip(zarr_image):
    uri, img, gt = zarr_image
    out = read_ome_zarr(uri)
    assert out.axes == "cyx"
    assert out.channel_names == ["DAPI", "CellMask"]
    np.testing.assert_array_equal(out.data, img)
    assert out.pixel_size_um == (0.65, 0.65)
    np.testing.assert_array_equal(read_labels(uri, "ground_truth"), gt)


def test_pyramid_and_ngff_metadata(zarr_image):
    uri, img, _ = zarr_image
    n = list_levels(uri)
    assert n >= 2
    low = read_ome_zarr(uri, level=1)
    assert low.data.shape == (2, img.shape[1] // 2, img.shape[2] // 2)
    assert low.pixel_size_um == (1.3, 1.3)
    root = zarr.open_group(uri, mode="r")
    assert root.attrs["multiscales"][0]["version"] == "0.4"
    assert root["labels"].attrs["labels"] == ["ground_truth"]


def test_read_image_tiff(tmp_path):
    import tifffile

    arr = np.random.default_rng(0).integers(0, 1000, (3, 32, 40), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "x.tif", arr)
    assert read_image(tmp_path / "x.tif").shape == (3, 32, 40)
    tifffile.imwrite(tmp_path / "y.tif", arr[0])
    assert read_image(tmp_path / "y.tif").shape == (1, 32, 40)
