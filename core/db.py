# core/db.py
from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Read from .env; fallback works for your local Docker container on port 3307
MYSQL_URL = os.getenv(
    "MYSQL_URL",
    "mysql+pymysql://root:pass123@127.0.0.1:3307/agentic_trader",
)

# Engine & session factory (SQLAlchemy 2.x style)
engine = create_engine(
    MYSQL_URL,
    pool_pre_ping=True,   # validate connections before using (handles MySQL idles)
    pool_recycle=3600,    # recycle connections hourly to avoid timeouts
    future=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


def init_db() -> None:
    """
    Create tables if they don't exist.
    Call this once on app start (e.g., in run_scheduler.py).
    """
    # import inside so model definitions are registered with Base
    from core import models  # noqa: F401
    Base.metadata.create_all(engine)


def get_session():
    """
    Small helper if you prefer context-managed sessions:
        with get_session() as s: ...
    """
    return SessionLocal()
