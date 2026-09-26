"""
Optimization Engine Tests (PuLP Linear Programming)
"""

import pytest
from backend.database import SessionLocal, init_db
from backend import schemas
from backend.services.optimization_service import optimize_reorder_quantities


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_pulp_reorder_optimization_bounds():
    db = SessionLocal()
    try:
        # Request with a tight budget and realistic warehouse capacity
        tight_budget = 5000.0
        req = schemas.OptimizationRequest(
            budget_limit=tight_budget,
            storage_limit=40000,
            service_level_z=1.65
        )
        res = optimize_reorder_quantities(db, req)

        assert res.optimization_status in ("Optimal", "Feasible")
        assert res.total_drugs_analyzed > 0
        assert res.total_estimated_cost <= tight_budget + 1e-4
        assert res.storage_utilized_units <= 40000

        # Check each recommendation item structure
        for rec in res.recommendations:
            assert rec.recommended_order_quantity >= 0
            assert rec.expected_inventory_after_order == rec.current_stock + rec.recommended_order_quantity
            assert rec.estimated_cost == round(rec.recommended_order_quantity * rec.unit_price, 2)
    finally:
        db.close()
