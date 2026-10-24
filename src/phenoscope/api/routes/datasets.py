from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from phenoscope.api.schemas import DatasetCreate, DatasetOut, ImageOut, ImageRegister
from phenoscope.db.models import Dataset, Image
from phenoscope.db.session import get_session
from phenoscope.io.omezarr import read_ome_zarr

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(body: DatasetCreate) -> DatasetOut:
    with get_session() as s:
        if s.scalar(select(Dataset).where(Dataset.name == body.name)):
            raise HTTPException(409, f"dataset {body.name!r} exists")
        ds = Dataset(**body.model_dump())
        s.add(ds)
        s.flush()
        return DatasetOut.model_validate(ds)


@router.get("", response_model=list[DatasetOut])
def list_datasets() -> list[DatasetOut]:
    with get_session() as s:
        counts = dict(s.execute(select(Image.dataset_id, func.count()).group_by(Image.dataset_id)).all())
        return [
            DatasetOut.model_validate(d).model_copy(update={"n_images": counts.get(d.id, 0)})
            for d in s.scalars(select(Dataset).order_by(Dataset.id))
        ]


@router.post("/{dataset_id}/images", response_model=ImageOut, status_code=201)
def register_image(dataset_id: int, body: ImageRegister) -> ImageOut:
    """Register an existing OME-Zarr image; dimensions/channels are read from its metadata."""
    with get_session() as s:
        if s.get(Dataset, dataset_id) is None:
            raise HTTPException(404, "dataset not found")
        try:
            img = read_ome_zarr(body.uri, level=0)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"cannot read OME-Zarr at {body.uri}: {exc}") from exc
        y, x = img.data.shape[img.axes.index("y")], img.data.shape[img.axes.index("x")]
        row = Image(
            dataset_id=dataset_id,
            width=x,
            height=y,
            channels={"names": img.channel_names},
            pixel_size_um=float(img.pixel_size_um[-1]) if img.pixel_size_um else 1.0,
            **body.model_dump(),
        )
        s.add(row)
        s.flush()
        return ImageOut.model_validate(row)


@router.get("/{dataset_id}/images", response_model=list[ImageOut])
def list_images(dataset_id: int, batch_id: str | None = None, plate: str | None = None):
    with get_session() as s:
        q = select(Image).where(Image.dataset_id == dataset_id)
        if batch_id:
            q = q.where(Image.batch_id == batch_id)
        if plate:
            q = q.where(Image.plate == plate)
        return [ImageOut.model_validate(i) for i in s.scalars(q.order_by(Image.id))]
