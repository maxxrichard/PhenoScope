"""``phenoscope`` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(help="PhenoScope: imaging AI for microscopy and digital pathology.",
                  no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _channels(s: str | None) -> list[str] | None:
    return [c.strip() for c in s.split(",")] if s else None


@app.command()
def convert(
    src: Path,
    dest: str = typer.Argument(..., help="Output OME-Zarr path or s3:// URI"),
    channels: str = typer.Option(None, help="Comma-separated channel names"),
    pixel_size: float = typer.Option(1.0, help="µm per pixel"),
    levels: int = 4,
) -> None:
    """Convert TIFF/PNG/NPY to a multiscale OME-Zarr store."""
    from phenoscope.io import read_image, write_ome_zarr

    data = read_image(src)
    write_ome_zarr(data, dest, axes="cyx", channel_names=_channels(channels),
                   pixel_size_um=(pixel_size, pixel_size), levels=levels, name=src.stem)
    typer.echo(f"wrote {dest} shape={data.shape}")


@app.command()
def segment(
    uri: str,
    segmenter: str = "watershed",
    channels: str = typer.Option(None, help="Channels to feed the model, e.g. DAPI,CellMask"),
    label_name: str = "cells",
    params: str = typer.Option("{}", help="JSON kwargs for the segmenter"),
) -> None:
    """Segment an OME-Zarr image and write labels/<label_name>."""
    from phenoscope.jobs.tasks import run_task

    res = run_task("segment", {"uri": uri, "segmenter": segmenter, "channels": _channels(channels),
                               "label_name": label_name, "params": json.loads(params)})
    typer.echo(json.dumps(res, indent=2, default=str))


@app.command()
def features(uri: str, out: str, label_name: str = "cells", texture: bool = False) -> None:
    """Extract per-object features to CSV/Parquet."""
    from phenoscope.jobs.tasks import run_task

    typer.echo(json.dumps(run_task("features", {"uri": uri, "out": out, "label_name": label_name,
                                                "texture": texture})))


@app.command()
def register(
    cycles: list[str] = typer.Argument(..., help="OME-Zarr per cycle, cycle 0 first"),
    out: str = typer.Option(..., help="Registered OME-Zarr output"),
    reference: str = "DAPI",
    affine: bool = True,
) -> None:
    """Register cyclic multiplex rounds on the nuclear reference channel and stack markers."""
    from phenoscope.io import read_ome_zarr, write_ome_zarr
    from phenoscope.registration import CycleImage, register_cycles

    imgs = [read_ome_zarr(u) for u in cycles]
    res = register_cycles([CycleImage(i.data, i.channel_names, reference) for i in imgs],
                          affine=affine)
    write_ome_zarr(res.image, out, channel_names=res.channel_names,
                   pixel_size_um=imgs[0].pixel_size_um[-2:],
                   extra_attrs={"registration": res.summary()})
    typer.echo(json.dumps(res.summary(), indent=2))


@app.command()
def benchmark(
    name: str = typer.Argument("synthetic", help="Benchmark name from configs/benchmarks.yaml"),
    models: str = typer.Option("configs/models.yaml", help="Segmenter configs to compare"),
    registry: Path = Path("configs/benchmarks.yaml"),
    baseline: str = "watershed",
    split: str = "test",
    out: Path | None = None,
) -> None:
    """Run segmenters on a curated benchmark and print AP@0.5 / mAP."""
    import yaml

    from phenoscope.benchmarks import InstanceDataset, load_registry, run_benchmark, synthetic_cells
    from phenoscope.benchmarks.runner import summarize

    cfg = yaml.safe_load(Path(models).read_text())["segmenters"]
    if name == "synthetic":
        pairs = [synthetic_cells(seed=s) for s in range(8)]
        cfg = {k: v for k, v in cfg.items() if v.get("backend", k) == "watershed"}
    else:
        pairs = InstanceDataset(load_registry(registry)[name], split=split)
    df = run_benchmark(pairs, cfg, dataset_name=name)
    typer.echo(summarize(df, baseline).round(3).to_string())
    if out:
        df.to_csv(out, index=False)


@app.command()
def drift(
    reference: Path = typer.Argument(..., help="CSV of reference per-image features"),
    batch: Path = typer.Argument(..., help="CSV of new batch per-image features"),
    batch_id: str = "batch",
) -> None:
    """Compare a new imaging batch to a benchmark reference; exit code 2 if drifted."""
    import pandas as pd

    from phenoscope.monitoring import DriftMonitor

    report = DriftMonitor(pd.read_csv(reference)).check(pd.read_csv(batch), batch_id=batch_id)
    for f in report.top(10):
        typer.echo(f"{f.feature:40s} PSI={f.psi:6.3f} KS={f.ks_stat:5.3f} [{f.level}]")
    typer.echo("DRIFT: " + "; ".join(report.reasons) if report.drifted else "no drift")
    raise typer.Exit(2 if report.drifted else 0)


@app.command("run-task")
def run_task_cmd(kind: str, payload: str, job_id: str = typer.Option(None)) -> None:
    """Entry point for worker pods: execute a task and report status to the metadata DB."""
    from phenoscope.jobs.backends import mark_job
    from phenoscope.jobs.tasks import run_task

    try:
        result = run_task(kind, json.loads(payload))
    except Exception as exc:
        if job_id:
            mark_job(job_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    if job_id:
        mark_job(job_id, "succeeded", result=result)
    typer.echo(json.dumps(result, default=str))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI server (API + web viewer)."""
    import uvicorn

    uvicorn.run("phenoscope.api.main:app", host=host, port=port, reload=reload)


@app.command()
def demo(root: Path = Path("data/demo")) -> None:
    """Create a small synthetic dataset (fluorescence, H&E, multiplex) and register it in the DB."""
    from phenoscope.benchmarks import synthetic_cells, synthetic_he
    from phenoscope.db import Dataset, Image, get_session, init_db
    from phenoscope.io import write_labels, write_ome_zarr

    init_db()
    root.mkdir(parents=True, exist_ok=True)
    with get_session() as s:
        ds = Dataset(name=f"demo-{root.name}", modality="fluorescence", uri=str(root),
                     description="Synthetic two-channel cells with ground truth")
        s.add(ds)
        s.flush()
        for i, well in enumerate(["A01", "A02", "B01", "B02"]):
            img, gt = synthetic_cells((512, 512), n_cells=120, seed=i)
            uri = str(root / f"plate1_{well}.zarr")
            write_ome_zarr(img, uri, channel_names=["DAPI", "CellMask"], pixel_size_um=(0.65, 0.65))
            write_labels(gt, uri, name="ground_truth")
            s.add(Image(dataset_id=ds.id, name=f"plate1_{well}", uri=uri, plate="plate1", well=well,
                        width=512, height=512, channels={"names": ["DAPI", "CellMask"]},
                        pixel_size_um=0.65, batch_id="batch-1"))
        he = synthetic_he((1024, 1024))
        he_uri = str(root / "he_slide.zarr")
        write_ome_zarr(he.transpose(2, 0, 1), he_uri, channel_names=["R", "G", "B"],
                       pixel_size_um=(0.5, 0.5))
        s.add(Image(dataset_id=ds.id, name="he_slide", uri=he_uri, width=1024, height=1024,
                    channels={"names": ["R", "G", "B"]}, pixel_size_um=0.5))
    typer.echo(f"demo dataset written to {root}")


if __name__ == "__main__":
    app()
