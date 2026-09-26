"""
Database Connection and Session Management
Supports MySQL / MariaDB (production) and SQLite (zero-config local dev/testing)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# Determine database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/drug_stock.db")

# Ensure local data directory exists if using sqlite
if DATABASE_URL.startswith("sqlite"):
    db_path = DATABASE_URL.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    # MySQL / MariaDB configuration with pooling and ping
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=10,
        max_overflow=20
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that provides a transactional database session per request.
    Rolls back automatically on unhandled exceptions and closes session.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Creates all defined tables if they do not already exist.
    """
    from backend import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
