"""
FastAPI Routes for Supplier Directory and Purchase Order Tracking
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import schemas, crud

router = APIRouter(tags=["Suppliers & Purchase Orders"])


@router.get("/suppliers", response_model=List[schemas.SupplierResponse])
def list_suppliers(db: Session = Depends(get_db)):
    """
    Retrieves all registered pharmaceutical vendors and lead times.
    """
    return crud.get_suppliers(db)


@router.post("/suppliers", response_model=schemas.SupplierResponse, status_code=status.HTTP_201_CREATED)
def add_new_supplier(
    sup_in: schemas.SupplierCreate,
    db: Session = Depends(get_db)
):
    """
    Registers a new supplier with contact details and standard lead time.
    """
    return crud.create_supplier(db, sup_in)


@router.get("/suppliers/{supplier_id}", response_model=schemas.SupplierResponse)
def get_supplier(
    supplier_id: int,
    db: Session = Depends(get_db)
):
    sup = crud.get_supplier(db, supplier_id)
    if not sup:
        raise HTTPException(status_code=404, detail=f"Supplier #{supplier_id} not found.")
    return sup


@router.get("/purchases", response_model=List[schemas.PurchaseResponse])
def list_purchase_orders(
    status: Optional[str] = Query(None, description="Pending, Delivered, Cancelled"),
    drug_id: Optional[int] = Query(None),
    supplier_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Retrieves procurement purchase orders.
    """
    return crud.get_purchases(db, status=status, drug_id=drug_id, supplier_id=supplier_id)


@router.post("/purchases", response_model=schemas.PurchaseResponse, status_code=status.HTTP_201_CREATED)
def create_purchase_order(
    pur_in: schemas.PurchaseCreate,
    db: Session = Depends(get_db)
):
    """
    Creates a new procurement purchase order for replenishment.
    """
    drug = crud.get_drug(db, pur_in.drug_id)
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug #{pur_in.drug_id} does not exist.")

    supplier = crud.get_supplier(db, pur_in.supplier_id)
    if not supplier:
        raise HTTPException(status_code=404, detail=f"Supplier #{pur_in.supplier_id} does not exist.")

    # Calculate expected delivery if not supplied
    if not pur_in.expected_delivery_date:
        from datetime import timedelta
        pur_in.expected_delivery_date = pur_in.purchase_date + timedelta(days=supplier.lead_time_days)

    purchase = crud.create_purchase(db, pur_in)
    unit_p = drug.unit_price

    return schemas.PurchaseResponse(
        purchase_id=purchase.purchase_id,
        drug_id=purchase.drug_id,
        drug_name=drug.drug_name,
        supplier_id=purchase.supplier_id,
        supplier_name=supplier.supplier_name,
        quantity=purchase.quantity,
        unit_price=unit_p,
        total_cost=round(unit_p * purchase.quantity, 2),
        purchase_date=purchase.purchase_date,
        expected_delivery_date=purchase.expected_delivery_date,
        status=purchase.status,
        created_at=purchase.created_at
    )


@router.put("/purchases/{purchase_id}/status", response_model=Dict[str, Any])
def change_purchase_status(
    purchase_id: int,
    status_update: schemas.PurchaseUpdate,
    db: Session = Depends(get_db)
):
    """
    Updates purchase order status.
    If marked as 'Delivered', automatically increments drug current_stock in inventory.
    """
    if not status_update.status:
        raise HTTPException(status_code=400, detail="Status field is required.")

    valid_statuses = {"Pending", "Delivered", "Cancelled"}
    if status_update.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")

    updated = crud.update_purchase_status(db, purchase_id, status_update.status)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Purchase order #{purchase_id} not found.")

    return {
        "message": f"Purchase #{purchase_id} status updated to {status_update.status}.",
        "purchase_id": purchase_id,
        "new_status": status_update.status
    }
