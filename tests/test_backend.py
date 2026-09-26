"""
Backend API Tests
"""

import pytest
from datetime import date, timedelta
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import SessionLocal, init_db
from backend import models

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()
    yield


def test_root_and_health():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "disclaimer" in data

    h_res = client.get("/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"


def test_drugs_api_lifecycle():
    unique_name = f"TestMed-{int(date.today().strftime('%Y%m%d%H%M%S'))}"
    payload = {
        "drug_name": unique_name,
        "category": "Antibiotics",
        "manufacturer": "Test Laboratories",
        "batch_number": "BAT-TEST-99",
        "expiry_date": (date.today() + timedelta(days=365)).strftime("%Y-%m-%d"),
        "unit_price": 25.50,
        "initial_stock": 150,
        "minimum_stock": 30,
        "maximum_stock": 600,
        "reorder_point": 50
    }

    # Create drug
    post_res = client.post("/drugs", json=payload)
    assert post_res.status_code == 201
    created_drug = post_res.json()
    drug_id = created_drug["drug_id"]
    assert created_drug["drug_name"] == unique_name
    assert created_drug["current_stock"] == 150

    # Get single drug
    get_res = client.get(f"/drugs/{drug_id}")
    assert get_res.status_code == 200
    assert get_res.json()["drug_id"] == drug_id

    # List drugs with search
    list_res = client.get(f"/drugs?search={unique_name}")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # Update drug
    update_res = client.put(f"/drugs/{drug_id}", json={"unit_price": 28.00})
    assert update_res.status_code == 200
    assert update_res.json()["unit_price"] == 28.00

    # Clean up
    del_res = client.delete(f"/drugs/{drug_id}")
    assert del_res.status_code == 200


def test_inventory_kpis():
    res = client.get("/inventory/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_drugs" in data
    assert "total_stock_units" in data
    assert "low_stock_items" in data
    assert "critical_items" in data
    assert "expired" in data


def test_inventory_expiring_breakdown():
    res = client.get("/inventory/expiring")
    assert res.status_code == 200
    data = res.json()
    assert "expired" in data
    assert "within_30_days" in data
    assert "within_60_days" in data
    assert "within_90_days" in data


def test_suppliers_and_purchases():
    # 1. Get suppliers
    s_res = client.get("/suppliers")
    assert s_res.status_code == 200
    suppliers = s_res.json()
    assert len(suppliers) > 0
    supplier_id = suppliers[0]["supplier_id"]

    # 2. Get first drug
    d_res = client.get("/drugs?limit=1")
    assert d_res.status_code == 200
    drugs = d_res.json()
    assert len(drugs) > 0
    drug_id = drugs[0]["drug_id"]

    # 3. Create purchase order
    purchase_payload = {
        "drug_id": drug_id,
        "supplier_id": supplier_id,
        "quantity": 100,
        "purchase_date": date.today().strftime("%Y-%m-%d"),
        "status": "Pending"
    }
    p_res = client.post("/purchases", json=purchase_payload)
    assert p_res.status_code == 201
    purchase = p_res.json()
    p_id = purchase["purchase_id"]
    assert purchase["status"] == "Pending"

    # 4. Update status to Delivered
    update_p = client.put(f"/purchases/{p_id}/status", json={"status": "Delivered"})
    assert update_p.status_code == 200
    assert update_p.json()["new_status"] == "Delivered"
