"""
FastAPI Routes for Drug Catalog Operations
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import schemas, crud

router = APIRouter(prefix="/drugs", tags=["Drugs"])


@router.get("", response_model=List[schemas.DrugResponse])
def list_drugs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search name, mfg, or batch"),
    db: Session = Depends(get_db)
):
    """
    Retrieves all drugs in catalog with optional category and search filters.
    """
    drugs, _ = crud.get_drugs(db, skip=skip, limit=limit, category=category, search=search)
    results = []
    for d in drugs:
        stock = d.inventory.current_stock if d.inventory else 0
        min_s = d.inventory.minimum_stock if d.inventory else 10
        max_s = d.inventory.maximum_stock if d.inventory else 500
        rop = d.inventory.reorder_point if d.inventory else 20
        st = d.inventory.calculate_status() if d.inventory else "NORMAL"
        results.append(schemas.DrugResponse(
            drug_id=d.drug_id,
            drug_name=d.drug_name,
            category=d.category,
            manufacturer=d.manufacturer,
            batch_number=d.batch_number,
            expiry_date=d.expiry_date,
            unit_price=d.unit_price,
            created_at=d.created_at,
            current_stock=stock,
            minimum_stock=min_s,
            maximum_stock=max_s,
            reorder_point=rop,
            status=st
        ))
    return results


@router.post("", response_model=schemas.DrugResponse, status_code=status.HTTP_201_CREATED)
def create_new_drug(
    drug_in: schemas.DrugCreate,
    db: Session = Depends(get_db)
):
    """
    Registers a new drug and creates its corresponding inventory record.
    """
    existing = crud.get_drug_by_name(db, drug_in.drug_name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Drug with name '{drug_in.drug_name}' already exists in catalog."
        )

    drug = crud.create_drug(db, drug_in)
    return schemas.DrugResponse(
        drug_id=drug.drug_id,
        drug_name=drug.drug_name,
        category=drug.category,
        manufacturer=drug.manufacturer,
        batch_number=drug.batch_number,
        expiry_date=drug.expiry_date,
        unit_price=drug.unit_price,
        created_at=drug.created_at,
        current_stock=drug.inventory.current_stock if drug.inventory else 0,
        minimum_stock=drug.inventory.minimum_stock if drug.inventory else 10,
        maximum_stock=drug.inventory.maximum_stock if drug.inventory else 500,
        reorder_point=drug.inventory.reorder_point if drug.inventory else 20,
        status="NORMAL"
    )


@router.get("/{drug_id}", response_model=schemas.DrugResponse)
def get_drug_details(
    drug_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieves full details for a single drug.
    """
    drug = crud.get_drug(db, drug_id)
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug with ID {drug_id} not found.")

    stock = drug.inventory.current_stock if drug.inventory else 0
    min_s = drug.inventory.minimum_stock if drug.inventory else 10
    max_s = drug.inventory.maximum_stock if drug.inventory else 500
    rop = drug.inventory.reorder_point if drug.inventory else 20
    st = drug.inventory.calculate_status() if drug.inventory else "NORMAL"

    return schemas.DrugResponse(
        drug_id=drug.drug_id,
        drug_name=drug.drug_name,
        category=drug.category,
        manufacturer=drug.manufacturer,
        batch_number=drug.batch_number,
        expiry_date=drug.expiry_date,
        unit_price=drug.unit_price,
        created_at=drug.created_at,
        current_stock=stock,
        minimum_stock=min_s,
        maximum_stock=max_s,
        reorder_point=rop,
        status=st
    )


@router.put("/{drug_id}", response_model=schemas.DrugResponse)
def update_existing_drug(
    drug_id: int,
    drug_in: schemas.DrugUpdate,
    db: Session = Depends(get_db)
):
    """
    Updates drug attributes.
    """
    drug = crud.update_drug(db, drug_id, drug_in)
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug with ID {drug_id} not found.")

    stock = drug.inventory.current_stock if drug.inventory else 0
    min_s = drug.inventory.minimum_stock if drug.inventory else 10
    max_s = drug.inventory.maximum_stock if drug.inventory else 500
    rop = drug.inventory.reorder_point if drug.inventory else 20
    st = drug.inventory.calculate_status() if drug.inventory else "NORMAL"

    return schemas.DrugResponse(
        drug_id=drug.drug_id,
        drug_name=drug.drug_name,
        category=drug.category,
        manufacturer=drug.manufacturer,
        batch_number=drug.batch_number,
        expiry_date=drug.expiry_date,
        unit_price=drug.unit_price,
        created_at=drug.created_at,
        current_stock=stock,
        minimum_stock=min_s,
        maximum_stock=max_s,
        reorder_point=rop,
        status=st
    )


@router.delete("/{drug_id}", status_code=status.HTTP_200_OK)
def delete_existing_drug(
    drug_id: int,
    db: Session = Depends(get_db)
):
    """
    Deletes a drug and cascades to inventory and historical records.
    """
    success = crud.delete_drug(db, drug_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Drug with ID {drug_id} not found.")
    return {"message": f"Drug #{drug_id} and associated records successfully removed."}
