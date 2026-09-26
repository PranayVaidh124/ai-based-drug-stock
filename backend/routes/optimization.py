"""
FastAPI Routes for Reorder Optimization using PuLP Linear Programming
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import schemas
from backend.services.optimization_service import optimize_reorder_quantities

router = APIRouter(prefix="/optimize-reorder", tags=["Reorder Optimization"])


@router.post("", response_model=schemas.OptimizationResponse)
def run_global_reorder_optimization(
    request: schemas.OptimizationRequest = schemas.OptimizationRequest(),
    db: Session = Depends(get_db)
):
    """
    Executes PuLP Linear Programming across all inventory items to compute
    cost-optimal reorder quantities subject to storage, budget, and priority constraints.
    """
    try:
        return optimize_reorder_quantities(db, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Linear Programming optimization error: {str(e)}")


@router.post("/{drug_id}", response_model=schemas.OptimizationResponse)
def run_single_drug_optimization(
    drug_id: int,
    request: schemas.OptimizationRequest = schemas.OptimizationRequest(),
    db: Session = Depends(get_db)
):
    """
    Executes PuLP optimization specifically for a single drug.
    """
    request.drug_ids = [drug_id]
    try:
        return optimize_reorder_quantities(db, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Linear Programming optimization error: {str(e)}")
