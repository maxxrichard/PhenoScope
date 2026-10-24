from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from phenoscope.config import get_settings
from phenoscope.db.models import Base


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {
        "pool_pre_ping": True, "pool_size": 10
    }
    return create_engine(url, **kwargs)


def init_db(url: str | None = None) -> Engine:
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(url: str | None = None) -> Iterator[Session]:
    session = sessionmaker(bind=get_engine(url), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
