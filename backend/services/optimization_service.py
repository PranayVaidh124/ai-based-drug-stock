"""
PuLP Linear Programming Optimization Engine for Drug Inventory Replenishment
Formulates a constrained Mixed-Integer Linear Program (MILP) to determine optimal
reorder quantities under budget, storage, lead-time, and priority constraints.
"""

from typing import List, Dict, Any, Optional
import pulp
import numpy as np
from sqlalchemy.orm import Session, joinedload

from backend import models, schemas, crud
from backend.services.forecasting_service import get_cached_model


def optimize_reorder_quantities(
    db: Session,
    request: schemas.OptimizationRequest
) -> schemas.OptimizationResponse:
    """
    Executes Linear Programming optimization using PuLP.
    Determines order quantities Q_i for candidate drugs subject to:
    1. Demand satisfaction: S_i + Q_i >= ROP_i + SafetyStock_i
    2. Capacity constraints: S_i + Q_i <= MaxStock_i
    3. Global Warehouse Storage Limit: sum(S_i + Q_i) <= StorageLimit
    4. Global Procurement Budget Limit: sum(Cost_i * Q_i) <= BudgetLimit
    5. Priority weighted penalties for stockout risk under constrained budgets.
    """
    budget_limit = request.budget_limit or 50000.0
    storage_limit = request.storage_limit or 20000
    service_level_z = request.service_level_z or 1.65

    # Query candidate drugs and inventory
    query = db.query(models.Drug).options(
        joinedload(models.Drug.inventory),
        joinedload(models.Drug.purchases).joinedload(models.Purchase.supplier)
    )

    if request.drug_ids and len(request.drug_ids) > 0:
        query = query.filter(models.Drug.drug_id.in_(request.drug_ids))

    all_drugs = query.all()
    if not all_drugs:
        raise ValueError("No drugs found matching the optimization criteria.")

    # Gather data for each drug
    candidate_items = []
    total_existing_units = 0

    for drug in all_drugs:
        inv = drug.inventory
        if not inv:
            continue

        current_stock = inv.current_stock
        total_existing_units += current_stock
        unit_price = drug.unit_price

        # Determine supplier lead time
        lead_time = request.lead_time_override or 7
        supplier_id = None
        supplier_name = "Primary Vendor"

        if drug.purchases and len(drug.purchases) > 0:
            latest_p = sorted(drug.purchases, key=lambda p: p.purchase_date, reverse=True)[0]
            if latest_p.supplier:
                lead_time = request.lead_time_override or latest_p.supplier.lead_time_days
                supplier_id = latest_p.supplier.supplier_id
                supplier_name = latest_p.supplier.supplier_name

        # Calculate consumption rate and demand over lead time
        consumptions = crud.get_consumption_for_drug(db, drug.drug_id, limit=90)
        if consumptions and len(consumptions) > 0:
            daily_rates = [c.quantity_consumed for c in consumptions]
            avg_daily = float(np.mean(daily_rates))
            std_daily = float(np.std(daily_rates)) if len(daily_rates) > 1 else avg_daily * 0.2
        else:
            avg_daily = 15.0
            std_daily = 3.5

        # Safety Stock and Reorder Point
        safety_stock = int(np.ceil(service_level_z * std_daily * np.sqrt(lead_time)))
        reorder_point = int(np.ceil((avg_daily * lead_time) + safety_stock))
        lead_time_demand = round(avg_daily * lead_time, 1)

        # Target stock level (bring stock back to safe operating ceiling)
        target_stock = max(inv.maximum_stock, reorder_point + safety_stock)
        status = inv.calculate_status()

        # Priority weight based on stock status urgency
        if status == "OUT OF STOCK":
            priority_weight = 1000.0
        elif status == "CRITICAL":
            priority_weight = 500.0
        elif status == "LOW STOCK":
            priority_weight = 200.0
        elif status in ("EXPIRED", "NEAR EXPIRY"):
            priority_weight = 150.0
        else:
            priority_weight = 10.0

        candidate_items.append({
            "drug_id": drug.drug_id,
            "drug_name": drug.drug_name,
            "category": drug.category,
            "unit_price": unit_price,
            "current_stock": current_stock,
            "minimum_stock": inv.minimum_stock,
            "maximum_stock": inv.maximum_stock,
            "reorder_point": reorder_point,
            "safety_stock": safety_stock,
            "predicted_demand": lead_time_demand,
            "lead_time_days": lead_time,
            "target_stock": target_stock,
            "status": status,
            "priority_weight": priority_weight,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name
        })

    # Available warehouse capacity
    remaining_storage_capacity = max(0, storage_limit - total_existing_units)

    # Initialize PuLP Linear Programming Problem
    # Goal: Maximize fulfilled target inventory weighted by medical urgency,
    # or equivalently minimize unfulfilled shortfall while respecting budget & space.
    prob = pulp.LpProblem("Drug_Reorder_Optimization", pulp.LpMinimize)

    # Decision Variables:
    # Q[i] = quantity to reorder for drug i (Integer >= 0)
    # Shortfall[i] = units below target stock if budget/space is constrained
    Q_vars = {}
    Shortfall_vars = {}

    for item in candidate_items:
        d_id = item["drug_id"]
        # Max reasonable order per item cannot exceed max stock minus current stock
        item_max_order = max(0, item["maximum_stock"] - item["current_stock"])

        Q_vars[d_id] = pulp.LpVariable(
            f"Q_{d_id}",
            lowBound=0,
            upBound=item_max_order,
            cat=pulp.LpInteger
        )
        Shortfall_vars[d_id] = pulp.LpVariable(
            f"Shortfall_{d_id}",
            lowBound=0,
            cat=pulp.LpContinuous
        )

    # Objective Function:
    # Minimize weighted shortfall + tiny penalty on purchase cost to avoid wasteful over-ordering
    prob += pulp.lpSum([
        (item["priority_weight"] * Shortfall_vars[item["drug_id"]]) +
        (0.001 * item["unit_price"] * Q_vars[item["drug_id"]])
        for item in candidate_items
    ])

    # Constraint 1: Budget Constraint (Total procurement cost <= Budget)
    prob += pulp.lpSum([
        item["unit_price"] * Q_vars[item["drug_id"]]
        for item in candidate_items
    ]) <= budget_limit, "Budget_Constraint"

    # Constraint 2: Global Storage Capacity Constraint
    prob += pulp.lpSum([
        Q_vars[item["drug_id"]]
        for item in candidate_items
    ]) <= remaining_storage_capacity, "Storage_Capacity_Constraint"

    # Constraint 3: Target Stock & Shortfall Definition for each drug
    # Current_Stock + Q + Shortfall >= Target_Stock
    for item in candidate_items:
        d_id = item["drug_id"]
        needed = max(0, item["target_stock"] - item["current_stock"])
        # If current stock is already above reorder point, no reorder needed unless low
        if item["current_stock"] > item["reorder_point"] and item["status"] not in ("EXPIRED", "NEAR EXPIRY"):
            prob += Q_vars[d_id] == 0, f"NoReorderNeeded_{d_id}"
            prob += Shortfall_vars[d_id] == 0, f"NoShortfall_{d_id}"
        else:
            prob += (Q_vars[d_id] + Shortfall_vars[d_id]) >= needed, f"TargetStockDef_{d_id}"

    # Solve optimization problem using CBC solver (bundled in PuLP)
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]

    # Collect recommendations
    recommendations: List[schemas.ReorderRecommendationItem] = []
    total_ordered_units = 0
    total_cost = 0.0
    reorder_count = 0

    for item in candidate_items:
        d_id = item["drug_id"]
        q_val = int(round(pulp.value(Q_vars[d_id]) or 0))
        q_val = max(0, q_val)

        expected_inv = item["current_stock"] + q_val
        cost = round(q_val * item["unit_price"], 2)

        if q_val > 0:
            reorder_count += 1
            total_ordered_units += q_val
            total_cost += cost

        recommendations.append(schemas.ReorderRecommendationItem(
            drug_id=item["drug_id"],
            drug_name=item["drug_name"],
            category=item["category"],
            unit_price=item["unit_price"],
            current_stock=item["current_stock"],
            minimum_stock=item["minimum_stock"],
            reorder_point=item["reorder_point"],
            safety_stock=item["safety_stock"],
            predicted_demand=item["predicted_demand"],
            lead_time_days=item["lead_time_days"],
            recommended_order_quantity=q_val,
            expected_inventory_after_order=expected_inv,
            estimated_cost=cost,
            stock_status=item["status"],
            supplier_id=item["supplier_id"],
            supplier_name=item["supplier_name"]
        ))

    # Sort recommendations by highest priority/cost
    recommendations.sort(key=lambda r: (r.recommended_order_quantity > 0, r.estimated_cost), reverse=True)

    budget_utilized_pct = round((total_cost / budget_limit) * 100, 2) if budget_limit > 0 else 0.0

    return schemas.OptimizationResponse(
        optimization_status=status_str,
        total_drugs_analyzed=len(candidate_items),
        drugs_to_reorder_count=reorder_count,
        total_estimated_cost=round(total_cost, 2),
        total_units_ordered=total_ordered_units,
        budget_limit=budget_limit,
        budget_utilized_percent=budget_utilized_pct,
        storage_limit=storage_limit,
        storage_utilized_units=total_existing_units + total_ordered_units,
        recommendations=recommendations
    )
