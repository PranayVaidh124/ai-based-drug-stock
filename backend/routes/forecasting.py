"""
FastAPI Routes for Demand Forecasting and Historical Consumption Analytics
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import schemas, crud
from backend.services.forecasting_service import generate_drug_forecast

router = APIRouter(tags=["Demand Forecasting & Consumption"])


@router.post("/forecast/{drug_id}", response_model=schemas.ForecastResponse)
def trigger_forecast(
    drug_id: int,
    request: schemas.ForecastRequest = schemas.ForecastRequest(days=30),
    db: Session = Depends(get_db)
):
    """
    Triggers Random Forest multi-step recursive demand prediction for the next 7 or 30 days.
    Calculates safety stock, reorder point, and model metrics.
    """
    try:
        return generate_drug_forecast(db, drug_id, days_ahead=request.days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting engine error: {str(e)}")


@router.get("/forecast/{drug_id}", response_model=schemas.ForecastResponse)
def get_cached_or_generate_forecast(
    drug_id: int,
    days: int = Query(30, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """
    Retrieves or calculates demand forecast for the designated drug.
    """
    try:
        return generate_drug_forecast(db, drug_id, days_ahead=days)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting service error: {str(e)}")


@router.get("/consumption/{drug_id}", response_model=List[schemas.ConsumptionResponse])
def get_drug_consumption_history(
    drug_id: int,
    limit: int = Query(365, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Retrieves daily historical consumption records for a specific drug.
    """
    records = crud.get_consumption_for_drug(db, drug_id, limit=limit)
    return records


@router.post("/consumption", response_model=schemas.ConsumptionResponse, status_code=status.HTTP_201_CREATED)
def log_consumption_entry(
    entry: schemas.ConsumptionCreate,
    db: Session = Depends(get_db)
):
    """
    Logs daily consumption and atomically decrements drug stock in inventory.
    """
    drug = crud.get_drug(db, entry.drug_id)
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug with ID {entry.drug_id} does not exist.")

    consumption = crud.record_consumption(db, entry)
    return consumption


@router.get("/analytics/monthly-consumption", response_model=List[Dict[str, Any]])
def get_monthly_consumption(db: Session = Depends(get_db)):
    """
    Aggregates historical consumption by year-month for trend visualization.
    """
    return crud.get_monthly_consumption_summary(db)


@router.get("/analytics/top-consumed", response_model=List[Dict[str, Any]])
def get_top_consumed(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Returns top most consumed drugs across all hospital wards/stores.
    """
    return crud.get_top_consumed_drugs(db, limit=limit)
