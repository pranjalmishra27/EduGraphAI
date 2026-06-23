"""
backend/core/database.py
Async SQLAlchemy engine, session factory, and base model.
Engine is lazily created to avoid import-time connection errors in tests.
"""
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from backend.core.config import get_settings

_engine = None
_AsyncSessionLocal = None


def _get_engine():
    global _engine, _AsyncSessionLocal
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url
        
        # Handle SQLite connection vs other databases
        if db_url.startswith("sqlite"):
            from pathlib import Path
            # Extract database path from sqlite+aiosqlite:///./data/edugraphai.db
            path_part = db_url.split("///")[-1]
            db_path = Path(path_part)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            _engine = create_async_engine(
                db_url,
                echo=settings.app_env == "development",
                connect_args={"timeout": 30},
            )
        else:
            _engine = create_async_engine(
                db_url,
                echo=settings.app_env == "development",
                pool_size=10,
                max_overflow=20,
            )
            
        _AsyncSessionLocal = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _engine, _AsyncSessionLocal


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    _, SessionLocal = _get_engine()
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    engine, _ = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
