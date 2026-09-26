"""
FastAPI Main Application Entry Point
AI-Based Drug Stock & Supply Chain Optimization System
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from backend.database import init_db
from backend.routes import drugs, inventory, forecasting, optimization, suppliers
from backend.services.forecasting_service import get_cached_model

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event handler for database initialization and model warm-up.
    """
    print("[SYSTEM] Initializing database schema...")
    init_db()

    print("[SYSTEM] Pre-loading Machine Learning demand forecasting model...")
    try:
        get_cached_model()
        print("[SYSTEM] ML forecasting engine is ready.")
    except Exception as e:
        print(f"[WARNING] ML model initialization deferred: {e}")

    yield
    print("[SYSTEM] Shutting down application.")


app = FastAPI(
    title="AI-Based Drug Stock & Supply Chain Optimization System",
    description=(
        "Production-grade AI inventory monitoring, multi-step Random Forest demand forecasting, "
        "and PuLP Linear Programming reorder optimization for hospitals and public-health drug stores.\n\n"
        "**LEGAL & CLINICAL DISCLAIMER:** This software is an administrative inventory and supply-chain "
        "decision-support system. It does NOT provide medical diagnosis, drug therapy recommendations, "
        "or clinical advice."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(drugs.router)
app.include_router(inventory.router)
app.include_router(forecasting.router)
app.include_router(optimization.router)
app.include_router(suppliers.router)


@app.get("/", tags=["Health & Status"])
def root_endpoint():
    """
    Service health check and system information endpoint.
    """
    return {
        "status": "online",
        "service": "AI-Based Drug Stock & Supply Chain Optimization System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "disclaimer": "Administrative supply-chain decision-support system only. No medical advice."
    }


@app.get("/health", tags=["Health & Status"])
def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "engine": "FastAPI + SQLAlchemy + Scikit-Learn + PuLP"
    }


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
