import time

from phenoscope.jobs.backends import KubernetesBackend


def _setup(client, uri):
    ds = client.post("/datasets", json={"name": "screen-1", "modality": "fluorescence",
                                        "uri": uri.rsplit("/", 1)[0]}).json()
    img = client.post(f"/datasets/{ds['id']}/images",
                      json={"name": "A01", "uri": uri, "plate": "plate1", "well": "A01",
                            "batch_id": "b1"}).json()
    return ds, img


def test_health_and_segmenters(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "watershed" in client.get("/segmenters").json()["available"]


def test_dataset_image_segment_features(client, zarr_image):
    uri, img_arr, gt = zarr_image
    ds, img = _setup(client, uri)
    assert img["width"] == 256 and img["channels"]["names"] == ["DAPI", "CellMask"]
    assert client.get("/datasets").json()[0]["n_images"] == 1
    assert client.post("/datasets", json={"name": "screen-1", "modality": "he",
                                          "uri": "x"}).status_code == 409

    r = client.post(f"/images/{img['id']}/segment", json={"segmenter": "watershed",
                                                          "channels": ["DAPI"]})
    assert r.status_code == 200, r.text
    assert abs(r.json()["n_objects"] - gt.max()) <= 2
    assert len(client.get(f"/images/{img['id']}/segmentations").json()) == 1

    feats = client.get(f"/images/{img['id']}/features").json()
    assert feats["n_objects"] == r.json()["n_objects"]
    assert "DAPI_intensity_mean" in feats["columns"]

    png = client.get(f"/images/{img['id']}/render.png", params={"labels": "cells"})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_async_job(client, zarr_image):
    uri, *_ = zarr_image
    _, img = _setup(client, uri)
    job = client.post(f"/images/{img['id']}/segment",
                      json={"segmenter": "watershed", "asynchronous": True}).json()
    for _ in range(100):
        st = client.get(f"/jobs/{job['id']}").json()
        if st["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert st["status"] == "succeeded", st
    assert st["result"]["n_objects"] > 0
    assert client.post("/jobs", json={"kind": "nope", "payload": {}}).status_code == 400


def test_drift_endpoint(client):
    ref = [{"DAPI_p99": 1000 + i} for i in range(50)]
    batch = [{"DAPI_p99": 1600 + i} for i in range(50)]
    r = client.post("/drift/check", json={"benchmark": "b", "batch_id": "x", "reference": ref,
                                          "batch": batch}).json()
    assert r["top_features"][0]["level"] == "alert"
    assert client.get("/drift/history").json()[0]["batch_id"] == "x"


def test_k8s_manifest_requests_gpu_for_segmentation():
    m = KubernetesBackend().manifest("abc", "segment", {"uri": "s3://b/x.zarr"})
    c = m["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert c["args"][:2] == ["run-task", "segment"]
    m2 = KubernetesBackend().manifest("abc", "features", {})
    assert "nvidia.com/gpu" not in m2["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
