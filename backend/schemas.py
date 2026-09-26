"""
Pydantic Schemas for Request & Response Data Validation
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ===================== DRUG SCHEMAS =====================

class DrugBase(BaseModel):
    drug_name: str = Field(..., min_length=2, max_length=150, description="Unique drug name")
    category: str = Field(..., min_length=2, max_length=100, description="Pharmacological category")
    manufacturer: Optional[str] = Field(None, max_length=150)
    batch_number: Optional[str] = Field(None, max_length=50)
    expiry_date: date = Field(..., description="Expiration date YYYY-MM-DD")
    unit_price: float = Field(..., gt=0, description="Unit cost/price in currency units")


class DrugCreate(DrugBase):
    initial_stock: int = Field(default=100, ge=0, description="Initial inventory stock")
    minimum_stock: int = Field(default=20, ge=0, description="Minimum safety floor")
    maximum_stock: int = Field(default=500, gt=0, description="Storage upper ceiling")
    reorder_point: int = Field(default=40, ge=0, description="Threshold to trigger reordering")


class DrugUpdate(BaseModel):
    drug_name: Optional[str] = Field(None, min_length=2, max_length=150)
    category: Optional[str] = Field(None, min_length=2, max_length=100)
    manufacturer: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[date] = None
    unit_price: Optional[float] = Field(None, gt=0)


class DrugResponse(DrugBase):
    drug_id: int
    created_at: Optional[datetime] = None
    current_stock: Optional[int] = 0
    minimum_stock: Optional[int] = 20
    maximum_stock: Optional[int] = 500
    reorder_point: Optional[int] = 40
    status: Optional[str] = "NORMAL"

    model_config = ConfigDict(from_attributes=True)


# ===================== INVENTORY SCHEMAS =====================

class InventoryBase(BaseModel):
    current_stock: int = Field(..., ge=0)
    minimum_stock: int = Field(..., ge=0)
    maximum_stock: int = Field(..., gt=0)
    reorder_point: int = Field(..., ge=0)


class InventoryUpdate(BaseModel):
    current_stock: Optional[int] = Field(None, ge=0)
    minimum_stock: Optional[int] = Field(None, ge=0)
    maximum_stock: Optional[int] = Field(None, gt=0)
    reorder_point: Optional[int] = Field(None, ge=0)
    stock_adjustment: Optional[int] = Field(None, description="Delta (+/-) to modify stock")


class InventoryResponse(BaseModel):
    inventory_id: int
    drug_id: int
    drug_name: str
    category: str
    manufacturer: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: date
    unit_price: float
    current_stock: int
    minimum_stock: int
    maximum_stock: int
    reorder_point: int
    last_updated: Optional[datetime] = None
    status: str

    model_config = ConfigDict(from_attributes=True)


# ===================== CONSUMPTION SCHEMAS =====================

class ConsumptionCreate(BaseModel):
    drug_id: int = Field(..., gt=0)
    date: date
    quantity_consumed: int = Field(..., ge=0)


class ConsumptionResponse(BaseModel):
    consumption_id: int
    drug_id: int
    date: date
    quantity_consumed: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ===================== SUPPLIER SCHEMAS =====================

class SupplierBase(BaseModel):
    supplier_name: str = Field(..., min_length=2, max_length=150)
    contact: Optional[str] = Field(None, max_length=100)
    lead_time_days: int = Field(..., gt=0, le=90, description="Delivery lead time in days")


class SupplierCreate(SupplierBase):
    pass


class SupplierResponse(SupplierBase):
    supplier_id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ===================== PURCHASE SCHEMAS =====================

class PurchaseCreate(BaseModel):
    drug_id: int = Field(..., gt=0)
    supplier_id: int = Field(..., gt=0)
    quantity: int = Field(..., gt=0)
    purchase_date: date = Field(default_factory=date.today)
    expected_delivery_date: Optional[date] = None
    status: str = Field(default="Pending")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {"Pending", "Delivered", "Cancelled"}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v


class PurchaseUpdate(BaseModel):
    status: Optional[str] = None
    quantity: Optional[int] = Field(None, gt=0)
    expected_delivery_date: Optional[date] = None


class PurchaseResponse(BaseModel):
    purchase_id: int
    drug_id: int
    drug_name: str
    supplier_id: int
    supplier_name: str
    quantity: int
    unit_price: float
    total_cost: float
    purchase_date: date
    expected_delivery_date: Optional[date] = None
    status: str
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ===================== FORECASTING SCHEMAS =====================

class ForecastRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=90, description="Forecast horizon: 7 or 30 days")


class ForecastPoint(BaseModel):
    date: str
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float


class ForecastResponse(BaseModel):
    drug_id: int
    drug_name: str
    category: str
    forecast_horizon_days: int
    total_predicted_demand: float
    average_daily_demand: float
    daily_demand_std: float
    metrics: Dict[str, Optional[float]]
    forecast_points: List[ForecastPoint]
    historical_points: List[Dict[str, Any]]
    reorder_point_calculated: int
    safety_stock_calculated: int
    current_stock: int
    status: str


# ===================== OPTIMIZATION SCHEMAS =====================

class OptimizationRequest(BaseModel):
    drug_ids: Optional[List[int]] = Field(None, description="Optional subset of drugs; empty = all drugs")
    budget_limit: Optional[float] = Field(default=50000.0, gt=0, description="Total budget in currency units")
    storage_limit: Optional[int] = Field(default=20000, gt=0, description="Total max unit storage capacity")
    service_level_z: Optional[float] = Field(default=1.65, gt=0, description="Z-score for safety stock (1.65 = 95%)")
    lead_time_override: Optional[int] = Field(None, gt=0, description="Optional global lead time override")


class ReorderRecommendationItem(BaseModel):
    drug_id: int
    drug_name: str
    category: str
    unit_price: float
    current_stock: int
    minimum_stock: int
    reorder_point: int
    safety_stock: int
    predicted_demand: float
    lead_time_days: int
    recommended_order_quantity: int
    expected_inventory_after_order: int
    estimated_cost: float
    stock_status: str
    supplier_id: Optional[int] = None
    supplier_name: Optional[str] = "Default Supplier"


class OptimizationResponse(BaseModel):
    optimization_status: str
    total_drugs_analyzed: int
    drugs_to_reorder_count: int
    total_estimated_cost: float
    total_units_ordered: int
    budget_limit: float
    budget_utilized_percent: float
    storage_limit: int
    storage_utilized_units: int
    recommendations: List[ReorderRecommendationItem]
