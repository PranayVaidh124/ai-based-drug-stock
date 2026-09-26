"""
FastAPI Routes for Inventory Monitoring, Stock Levels, and Expiry Alerts
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import schemas, crud

router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.get("", response_model=List[schemas.InventoryResponse])
def list_inventory(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    status_filter: Optional[str] = Query(None, description="NORMAL, LOW STOCK, CRITICAL, OUT OF STOCK, NEAR EXPIRY, EXPIRED"),
    category: Optional[str] = Query(None, description="Filter category"),
    search: Optional[str] = Query(None, description="Search term"),
    db: Session = Depends(get_db)
):
    """
    Returns inventory items with calculated status, current stock, and thresholds.
    """
    items, _ = crud.get_inventory_items(
        db,
        skip=skip,
        limit=limit,
        status_filter=status_filter,
        category_filter=category,
        search=search
    )
    return items


@router.get("/summary", response_model=Dict[str, Any])
def get_kpi_summary(db: Session = Depends(get_db)):
    """
    Returns executive KPI card metrics:
    - total_drugs, total_stock_units, low_stock_items, critical_items, out_of_stock, near_expiry, expired
    """
    return crud.get_inventory_summary_kpis(db)


@router.get("/low-stock", response_model=List[schemas.InventoryResponse])
def get_low_stock(db: Session = Depends(get_db)):
    """
    Returns items currently at or below their reorder point.
    """
    items, _ = crud.get_inventory_items(db, limit=500, status_filter="LOW STOCK")
    return items


@router.get("/critical", response_model=List[schemas.InventoryResponse])
def get_critical_stock(db: Session = Depends(get_db)):
    """
    Returns items currently at or below minimum safety floor.
    """
    items, _ = crud.get_inventory_items(db, limit=500, status_filter="CRITICAL")
    return items


@router.get("/out-of-stock", response_model=List[schemas.InventoryResponse])
def get_out_of_stock(db: Session = Depends(get_db)):
    """
    Returns items with zero stock units.
    """
    items, _ = crud.get_inventory_items(db, limit=500, status_filter="OUT OF STOCK")
    return items


@router.get("/expired", response_model=List[Dict[str, Any]])
def get_expired_items(db: Session = Depends(get_db)):
    """
    Returns items whose expiry date has passed.
    """
    buckets = crud.get_expiry_buckets(db)
    return buckets["expired"]


@router.get("/expiring", response_model=Dict[str, List[Dict[str, Any]]])
def get_expiring_breakdown(db: Session = Depends(get_db)):
    """
    Returns segmented expiry alert buckets:
    - expired
    - within_30_days
    - within_60_days
    - within_90_days
    """
    return crud.get_expiry_buckets(db)


@router.get("/{drug_id}", response_model=schemas.InventoryResponse)
def get_single_inventory(
    drug_id: int,
    db: Session = Depends(get_db)
):
    inv = crud.get_inventory_by_drug_id(db, drug_id)
    if not inv or not inv.drug:
        raise HTTPException(status_code=404, detail=f"Inventory record for drug #{drug_id} not found.")

    return schemas.InventoryResponse(
        inventory_id=inv.inventory_id,
        drug_id=inv.drug_id,
        drug_name=inv.drug.drug_name,
        category=inv.drug.category,
        manufacturer=inv.drug.manufacturer,
        batch_number=inv.drug.batch_number,
        expiry_date=inv.drug.expiry_date,
        unit_price=inv.drug.unit_price,
        current_stock=inv.current_stock,
        minimum_stock=inv.minimum_stock,
        maximum_stock=inv.maximum_stock,
        reorder_point=inv.reorder_point,
        last_updated=inv.last_updated,
        status=inv.calculate_status()
    )


@router.put("/{drug_id}", response_model=schemas.InventoryResponse)
def update_stock_levels(
    drug_id: int,
    inv_in: schemas.InventoryUpdate,
    db: Session = Depends(get_db)
):
    """
    Updates inventory stock limits or applies a stock adjustment.
    """
    inv = crud.update_inventory(db, drug_id, inv_in)
    if not inv or not inv.drug:
        raise HTTPException(status_code=404, detail=f"Inventory record for drug #{drug_id} not found.")

    return schemas.InventoryResponse(
        inventory_id=inv.inventory_id,
        drug_id=inv.drug_id,
        drug_name=inv.drug.drug_name,
        category=inv.drug.category,
        manufacturer=inv.drug.manufacturer,
        batch_number=inv.drug.batch_number,
        expiry_date=inv.drug.expiry_date,
        unit_price=inv.drug.unit_price,
        current_stock=inv.current_stock,
        minimum_stock=inv.minimum_stock,
        maximum_stock=inv.maximum_stock,
        reorder_point=inv.reorder_point,
        last_updated=inv.last_updated,
        status=inv.calculate_status()
    )
