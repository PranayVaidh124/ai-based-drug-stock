"""
CRUD Operations for Drug Inventory, Suppliers, Purchases, Consumption, and Forecasts
"""

from datetime import date, timedelta
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, and_, func

from backend import models, schemas


# ===================== DRUG CRUD =====================

def get_drugs(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> Tuple[List[models.Drug], int]:
    query = db.query(models.Drug).options(joinedload(models.Drug.inventory))
    if category and category.lower() != "all":
        query = query.filter(models.Drug.category.ilike(f"%{category}%"))
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            or_(
                models.Drug.drug_name.ilike(search_pattern),
                models.Drug.manufacturer.ilike(search_pattern),
                models.Drug.batch_number.ilike(search_pattern)
            )
        )
    total = query.count()
    items = query.order_by(models.Drug.drug_name.asc()).offset(skip).limit(limit).all()
    return items, total


def get_drug(db: Session, drug_id: int) -> Optional[models.Drug]:
    return db.query(models.Drug).options(
        joinedload(models.Drug.inventory),
        joinedload(models.Drug.purchases)
    ).filter(models.Drug.drug_id == drug_id).first()


def get_drug_by_name(db: Session, drug_name: str) -> Optional[models.Drug]:
    return db.query(models.Drug).filter(models.Drug.drug_name.ilike(drug_name.strip())).first()


def create_drug(db: Session, drug_in: schemas.DrugCreate) -> models.Drug:
    """
    Creates drug and initial inventory record in a single transaction.
    """
    drug = models.Drug(
        drug_name=drug_in.drug_name.strip(),
        category=drug_in.category.strip(),
        manufacturer=drug_in.manufacturer.strip() if drug_in.manufacturer else None,
        batch_number=drug_in.batch_number.strip() if drug_in.batch_number else None,
        expiry_date=drug_in.expiry_date,
        unit_price=float(drug_in.unit_price)
    )
    db.add(drug)
    db.flush()  # Allocates drug.drug_id

    inventory = models.Inventory(
        drug_id=drug.drug_id,
        current_stock=drug_in.initial_stock,
        minimum_stock=drug_in.minimum_stock,
        maximum_stock=drug_in.maximum_stock,
        reorder_point=drug_in.reorder_point
    )
    db.add(inventory)
    db.commit()
    db.refresh(drug)
    return drug


def update_drug(db: Session, drug_id: int, drug_in: schemas.DrugUpdate) -> Optional[models.Drug]:
    drug = get_drug(db, drug_id)
    if not drug:
        return None
    data = drug_in.model_dump(exclude_unset=True)
    for field, val in data.items():
        if val is not None:
            setattr(drug, field, val)
    db.commit()
    db.refresh(drug)
    return drug


def delete_drug(db: Session, drug_id: int) -> bool:
    drug = get_drug(db, drug_id)
    if not drug:
        return False
    db.delete(drug)
    db.commit()
    return True


# ===================== INVENTORY CRUD =====================

def get_inventory_items(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    search: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int]:
    query = db.query(models.Inventory).join(models.Drug).options(joinedload(models.Inventory.drug))

    if category_filter and category_filter.lower() != "all":
        query = query.filter(models.Drug.category.ilike(f"%{category_filter}%"))

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            or_(
                models.Drug.drug_name.ilike(search_pattern),
                models.Drug.manufacturer.ilike(search_pattern),
                models.Drug.batch_number.ilike(search_pattern)
            )
        )

    all_matches = query.all()
    today = date.today()

    # Calculate status and apply status filter
    formatted = []
    for inv in all_matches:
        st = inv.calculate_status(today)
        if status_filter and status_filter.upper() != "ALL":
            if st != status_filter.upper():
                continue

        formatted.append({
            "inventory_id": inv.inventory_id,
            "drug_id": inv.drug_id,
            "drug_name": inv.drug.drug_name,
            "category": inv.drug.category,
            "manufacturer": inv.drug.manufacturer,
            "batch_number": inv.drug.batch_number,
            "expiry_date": inv.drug.expiry_date,
            "unit_price": inv.drug.unit_price,
            "current_stock": inv.current_stock,
            "minimum_stock": inv.minimum_stock,
            "maximum_stock": inv.maximum_stock,
            "reorder_point": inv.reorder_point,
            "last_updated": inv.last_updated,
            "status": st
        })

    total = len(formatted)
    paginated = formatted[skip: skip + limit]
    return paginated, total


def get_inventory_by_drug_id(db: Session, drug_id: int) -> Optional[models.Inventory]:
    return db.query(models.Inventory).filter(models.Inventory.drug_id == drug_id).first()


def update_inventory(db: Session, drug_id: int, inv_in: schemas.InventoryUpdate) -> Optional[models.Inventory]:
    inventory = get_inventory_by_drug_id(db, drug_id)
    if not inventory:
        return None

    if inv_in.current_stock is not None:
        inventory.current_stock = inv_in.current_stock
    if inv_in.stock_adjustment is not None:
        inventory.current_stock = max(0, inventory.current_stock + inv_in.stock_adjustment)
    if inv_in.minimum_stock is not None:
        inventory.minimum_stock = inv_in.minimum_stock
    if inv_in.maximum_stock is not None:
        inventory.maximum_stock = inv_in.maximum_stock
    if inv_in.reorder_point is not None:
        inventory.reorder_point = inv_in.reorder_point

    db.commit()
    db.refresh(inventory)
    return inventory


def get_inventory_summary_kpis(db: Session) -> Dict[str, Any]:
    inventories = db.query(models.Inventory).options(joinedload(models.Inventory.drug)).all()
    today = date.today()

    total_drugs = len(inventories)
    total_stock_units = 0
    low_stock_items = 0
    critical_items = 0
    out_of_stock = 0
    near_expiry = 0
    expired = 0

    category_counts: Dict[str, int] = {}
    category_stock: Dict[str, int] = {}

    for inv in inventories:
        drug = inv.drug
        stock = inv.current_stock
        total_stock_units += stock

        if drug:
            category_counts[drug.category] = category_counts.get(drug.category, 0) + 1
            category_stock[drug.category] = category_stock.get(drug.category, 0) + stock

            days_to_expiry = (drug.expiry_date - today).days
            if days_to_expiry < 0:
                expired += 1
            elif days_to_expiry <= 30:
                near_expiry += 1

        if stock <= 0:
            out_of_stock += 1
        elif stock <= inv.minimum_stock:
            critical_items += 1
        elif stock <= inv.reorder_point:
            low_stock_items += 1

    return {
        "total_drugs": total_drugs,
        "total_stock_units": total_stock_units,
        "low_stock_items": low_stock_items,
        "critical_items": critical_items,
        "out_of_stock": out_of_stock,
        "near_expiry": near_expiry,
        "expired": expired,
        "category_counts": category_counts,
        "category_stock": category_stock
    }


def get_expiry_buckets(db: Session) -> Dict[str, List[Dict[str, Any]]]:
    today = date.today()
    drugs = db.query(models.Drug).options(joinedload(models.Drug.inventory)).all()

    buckets = {
        "expired": [],
        "within_30_days": [],
        "within_60_days": [],
        "within_90_days": []
    }

    for drug in drugs:
        days = (drug.expiry_date - today).days
        item = {
            "drug_id": drug.drug_id,
            "drug_name": drug.drug_name,
            "category": drug.category,
            "batch_number": drug.batch_number,
            "expiry_date": drug.expiry_date,
            "days_until_expiry": days,
            "current_stock": drug.inventory.current_stock if drug.inventory else 0,
            "unit_price": drug.unit_price,
            "loss_value": (drug.inventory.current_stock if drug.inventory else 0) * drug.unit_price
        }

        if days < 0:
            buckets["expired"].append(item)
        elif days <= 30:
            buckets["within_30_days"].append(item)
        elif days <= 60:
            buckets["within_60_days"].append(item)
        elif days <= 90:
            buckets["within_90_days"].append(item)

    return buckets


# ===================== CONSUMPTION CRUD =====================

def record_consumption(db: Session, con_in: schemas.ConsumptionCreate) -> models.Consumption:
    """
    Records daily consumption and atomically decrements inventory current_stock.
    """
    consumption = models.Consumption(
        drug_id=con_in.drug_id,
        date=con_in.date,
        quantity_consumed=con_in.quantity_consumed
    )
    db.add(consumption)

    inv = get_inventory_by_drug_id(db, con_in.drug_id)
    if inv:
        inv.current_stock = max(0, inv.current_stock - con_in.quantity_consumed)

    db.commit()
    db.refresh(consumption)
    return consumption


def get_consumption_for_drug(db: Session, drug_id: int, limit: int = 365) -> List[models.Consumption]:
    return db.query(models.Consumption).filter(
        models.Consumption.drug_id == drug_id
    ).order_by(models.Consumption.date.asc()).limit(limit).all()


def get_monthly_consumption_summary(db: Session) -> List[Dict[str, Any]]:
    """
    Summarizes consumption by month for dashboard charts.
    """
    records = db.query(
        models.Consumption.date,
        models.Consumption.quantity_consumed
    ).all()

    monthly: Dict[str, int] = {}
    for d, q in records:
        key = d.strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0) + q

    sorted_res = [{"month": k, "quantity": v} for k, v in sorted(monthly.items())]
    return sorted_res


def get_top_consumed_drugs(db: Session, limit: int = 10) -> List[Dict[str, Any]]:
    results = db.query(
        models.Drug.drug_name,
        models.Drug.category,
        func.sum(models.Consumption.quantity_consumed).label("total_consumed")
    ).join(models.Consumption, models.Drug.drug_id == models.Consumption.drug_id)\
     .group_by(models.Drug.drug_id, models.Drug.drug_name, models.Drug.category)\
     .order_by(func.sum(models.Consumption.quantity_consumed).desc())\
     .limit(limit).all()

    return [
        {"drug_name": r[0], "category": r[1], "total_consumed": int(r[2] or 0)}
        for r in results
    ]


# ===================== SUPPLIER CRUD =====================

def get_suppliers(db: Session) -> List[models.Supplier]:
    return db.query(models.Supplier).order_by(models.Supplier.supplier_name.asc()).all()


def get_supplier(db: Session, supplier_id: int) -> Optional[models.Supplier]:
    return db.query(models.Supplier).filter(models.Supplier.supplier_id == supplier_id).first()


def create_supplier(db: Session, sup_in: schemas.SupplierCreate) -> models.Supplier:
    supplier = models.Supplier(
        supplier_name=sup_in.supplier_name.strip(),
        contact=sup_in.contact.strip() if sup_in.contact else None,
        lead_time_days=sup_in.lead_time_days
    )
    db.add(supplier)
    db.commit()
    db.refresh(supplier)
    return supplier


# ===================== PURCHASE CRUD =====================

def get_purchases(
    db: Session,
    status: Optional[str] = None,
    drug_id: Optional[int] = None,
    supplier_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    query = db.query(models.Purchase).options(
        joinedload(models.Purchase.drug),
        joinedload(models.Purchase.supplier)
    )
    if status and status.lower() != "all":
        query = query.filter(models.Purchase.status.ilike(status))
    if drug_id:
        query = query.filter(models.Purchase.drug_id == drug_id)
    if supplier_id:
        query = query.filter(models.Purchase.supplier_id == supplier_id)

    purchases = query.order_by(models.Purchase.purchase_date.desc()).all()
    results = []
    for p in purchases:
        unit_price = p.drug.unit_price if p.drug else 0.0
        results.append({
            "purchase_id": p.purchase_id,
            "drug_id": p.drug_id,
            "drug_name": p.drug.drug_name if p.drug else f"Drug #{p.drug_id}",
            "supplier_id": p.supplier_id,
            "supplier_name": p.supplier.supplier_name if p.supplier else f"Supplier #{p.supplier_id}",
            "quantity": p.quantity,
            "unit_price": unit_price,
            "total_cost": round(unit_price * p.quantity, 2),
            "purchase_date": p.purchase_date,
            "expected_delivery_date": p.expected_delivery_date,
            "status": p.status,
            "created_at": p.created_at
        })
    return results


def create_purchase(db: Session, pur_in: schemas.PurchaseCreate) -> models.Purchase:
    purchase = models.Purchase(
        drug_id=pur_in.drug_id,
        supplier_id=pur_in.supplier_id,
        quantity=pur_in.quantity,
        purchase_date=pur_in.purchase_date,
        expected_delivery_date=pur_in.expected_delivery_date,
        status=pur_in.status
    )
    db.add(purchase)
    db.commit()
    db.refresh(purchase)
    return purchase


def update_purchase_status(db: Session, purchase_id: int, new_status: str) -> Optional[models.Purchase]:
    purchase = db.query(models.Purchase).filter(models.Purchase.purchase_id == purchase_id).first()
    if not purchase:
        return None

    prev_status = purchase.status
    purchase.status = new_status

    # If marked as Delivered from Pending, increment inventory
    if new_status.lower() == "delivered" and prev_status.lower() != "delivered":
        inv = get_inventory_by_drug_id(db, purchase.drug_id)
        if inv:
            inv.current_stock += purchase.quantity

    db.commit()
    db.refresh(purchase)
    return purchase
