"""
SQLAlchemy ORM Models for Drug Stock & Supply Chain Management System
"""

from datetime import datetime, date
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    Index
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.database import Base


class Drug(Base):
    __tablename__ = "drugs"

    drug_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drug_name = Column(String(150), unique=True, nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    manufacturer = Column(String(150), nullable=True)
    batch_number = Column(String(50), nullable=True)
    expiry_date = Column(Date, nullable=False, index=True)
    unit_price = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    inventory = relationship("Inventory", uselist=False, back_populates="drug", cascade="all, delete-orphan")
    consumptions = relationship("Consumption", back_populates="drug", cascade="all, delete-orphan")
    purchases = relationship("Purchase", back_populates="drug")
    forecasts = relationship("Forecast", back_populates="drug", cascade="all, delete-orphan")


class Inventory(Base):
    __tablename__ = "inventory"

    inventory_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey("drugs.drug_id", ondelete="CASCADE"), unique=True, nullable=False)
    current_stock = Column(Integer, nullable=False, default=0)
    minimum_stock = Column(Integer, nullable=False, default=10)
    maximum_stock = Column(Integer, nullable=False, default=500)
    reorder_point = Column(Integer, nullable=False, default=20)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship
    drug = relationship("Drug", back_populates="inventory")

    def calculate_status(self, target_date: date = None) -> str:
        """
        Classifies stock status into:
        - EXPIRED
        - NEAR EXPIRY (<= 30 days)
        - OUT OF STOCK (current_stock <= 0)
        - CRITICAL (current_stock <= minimum_stock)
        - LOW STOCK (current_stock <= reorder_point)
        - NORMAL
        """
        now_date = target_date or date.today()
        if self.drug and self.drug.expiry_date:
            days_to_expiry = (self.drug.expiry_date - now_date).days
            if days_to_expiry < 0:
                return "EXPIRED"
            elif days_to_expiry <= 30:
                return "NEAR EXPIRY"

        if self.current_stock <= 0:
            return "OUT OF STOCK"
        elif self.current_stock <= self.minimum_stock:
            return "CRITICAL"
        elif self.current_stock <= self.reorder_point:
            return "LOW STOCK"
        return "NORMAL"


class Consumption(Base):
    __tablename__ = "consumption"

    consumption_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey("drugs.drug_id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    quantity_consumed = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    drug = relationship("Drug", back_populates="consumptions")

    __table_args__ = (
        Index("idx_consumption_drug_date", "drug_id", "date"),
    )


class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    supplier_name = Column(String(150), nullable=False, index=True)
    contact = Column(String(100), nullable=True)
    lead_time_days = Column(Integer, nullable=False, default=7)
    created_at = Column(DateTime, default=func.now())

    # Relationship
    purchases = relationship("Purchase", back_populates="supplier")


class Purchase(Base):
    __tablename__ = "purchases"

    purchase_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey("drugs.drug_id"), nullable=False, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.supplier_id"), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    purchase_date = Column(Date, nullable=False)
    expected_delivery_date = Column(Date, nullable=True)
    status = Column(String(50), nullable=False, default="Pending")
    created_at = Column(DateTime, default=func.now())

    # Relationships
    drug = relationship("Drug", back_populates="purchases")
    supplier = relationship("Supplier", back_populates="purchases")


class Forecast(Base):
    __tablename__ = "forecasts"

    forecast_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey("drugs.drug_id", ondelete="CASCADE"), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    predicted_demand = Column(Float, nullable=False, default=0.0)
    confidence_level = Column(String(50), nullable=False, default="Normal")
    created_at = Column(DateTime, default=func.now())

    # Relationship
    drug = relationship("Drug", back_populates="forecasts")

    __table_args__ = (
        Index("idx_forecasts_drug_date", "drug_id", "forecast_date"),
    )
