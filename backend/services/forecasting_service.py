"""
Forecasting Service: Demand Prediction, Safety Stock, and Dynamic Reorder Point Calculation
"""

import os
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend import models, schemas, crud


MODEL_PATH = os.getenv("MODEL_PATH", "ml/model.pkl")
_MODEL_CACHE: Optional[Dict[str, Any]] = None


def get_cached_model():
    """
    Loads and caches the Random Forest model bundle in memory.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            from ml.train_model import train_forecasting_model
            _MODEL_CACHE = train_forecasting_model(model_output_path=MODEL_PATH)
        else:
            _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def generate_drug_forecast(
    db: Session,
    drug_id: int,
    days_ahead: int = 30,
    service_level_z: float = 1.65
) -> schemas.ForecastResponse:
    """
    Performs multi-step recursive demand forecasting for a specific drug.
    Computes:
    - Step-by-step future predictions with confidence bounds
    - Dynamic safety stock: SS = Z * sigma_demand * sqrt(lead_time)
    - Dynamic reorder point: ROP = Average_Daily_Demand * lead_time + SS
    - Stores forecast points into database for audit and history
    """
    drug = crud.get_drug(db, drug_id)
    if not drug:
        raise ValueError(f"Drug with ID {drug_id} does not exist.")

    # Retrieve historical consumption for the drug
    history_records = crud.get_consumption_for_drug(db, drug_id, limit=365)
    if not history_records or len(history_records) < 14:
        raise ValueError(f"Insufficient historical data for drug #{drug_id}. At least 14 days required.")

    # Prepare historical series
    dates = [rec.date for rec in history_records]
    quantities = [float(rec.quantity_consumed) for rec in history_records]
    hist_df = pd.DataFrame({"date": pd.to_datetime(dates), "quantity": quantities}).sort_values("date")

    avg_daily_demand = float(hist_df["quantity"].mean())
    daily_demand_std = float(hist_df["quantity"].std() if len(hist_df) > 1 else avg_daily_demand * 0.25)
    if pd.isna(daily_demand_std) or daily_demand_std == 0:
        daily_demand_std = max(1.0, avg_daily_demand * 0.2)

    # Lead time determination (from latest supplier or default 7 days)
    lead_time = 7
    if drug.purchases and len(drug.purchases) > 0:
        latest_purchase = sorted(drug.purchases, key=lambda p: p.purchase_date, reverse=True)[0]
        if latest_purchase.supplier:
            lead_time = latest_purchase.supplier.lead_time_days

    # Safety stock and Reorder Point formula
    safety_stock = int(np.ceil(service_level_z * daily_demand_std * np.sqrt(lead_time)))
    reorder_point = int(np.ceil((avg_daily_demand * lead_time) + safety_stock))

    # Load ML Model
    bundle = get_cached_model()
    rf_model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    category_map = bundle["category_map"]
    metrics = bundle.get("metrics", {})

    category_code = category_map.get(drug.category, 0)

    # Multi-step recursive forecasting
    # We maintain an extended list of past and forecasted quantities
    q_history = list(hist_df["quantity"].values)
    last_date = hist_df["date"].iloc[-1].date()

    forecast_points: List[schemas.ForecastPoint] = []
    total_forecasted = 0.0

    # Delete previous forecasts for this drug to prevent duplicate stacking
    db.query(models.Forecast).filter(models.Forecast.drug_id == drug_id).delete()

    db_forecast_records = []

    for step in range(1, days_ahead + 1):
        target_date = last_date + timedelta(days=step)
        month = target_date.month
        day = target_date.day
        day_of_week = target_date.weekday()
        is_weekend = 1 if day_of_week in (5, 6) else 0
        quarter = (month - 1) // 3 + 1

        # Lags from the growing q_history list
        lag_1 = q_history[-1]
        lag_7 = q_history[-7] if len(q_history) >= 7 else avg_daily_demand
        lag_14 = q_history[-14] if len(q_history) >= 14 else avg_daily_demand
        lag_30 = q_history[-30] if len(q_history) >= 30 else avg_daily_demand

        # Rolling statistics
        r_slice_7 = q_history[-7:]
        r_slice_14 = q_history[-14:]
        r_slice_30 = q_history[-30:]

        rolling_mean_7 = float(np.mean(r_slice_7))
        rolling_std_7 = float(np.std(r_slice_7)) if len(r_slice_7) > 1 else 0.0
        rolling_mean_14 = float(np.mean(r_slice_14))
        rolling_mean_30 = float(np.mean(r_slice_30))

        feat_dict = {
            "category_code": category_code,
            "month": month,
            "day": day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "quarter": quarter,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_30": lag_30,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "rolling_mean_14": rolling_mean_14,
            "rolling_mean_30": rolling_mean_30
        }

        # Create feature vector matching model's expected column sequence
        X_step = np.array([[feat_dict[col] for col in feature_cols]])

        # Estimate individual trees variance for confidence bounds
        tree_preds = [tree.predict(X_step)[0] for tree in rf_model.estimators_]
        pred_val = float(np.mean(tree_preds))
        pred_val = max(0.0, pred_val)

        pred_std = float(np.std(tree_preds))
        lower_bound = max(0.0, pred_val - (service_level_z * pred_std))
        upper_bound = pred_val + (service_level_z * pred_std)

        # Append to recursive history
        q_history.append(pred_val)
        total_forecasted += pred_val

        f_point = schemas.ForecastPoint(
            date=target_date.strftime("%Y-%m-%d"),
            predicted_demand=round(pred_val, 2),
            confidence_lower=round(lower_bound, 2),
            confidence_upper=round(upper_bound, 2)
        )
        forecast_points.append(f_point)

        # Confidence / risk level classification
        risk_level = "High Demand Risk" if pred_val > (avg_daily_demand * 1.3) else "Normal"
        f_record = models.Forecast(
            drug_id=drug_id,
            forecast_date=target_date,
            predicted_demand=round(pred_val, 2),
            confidence_level=risk_level
        )
        db_forecast_records.append(f_record)

    # Commit forecasts to DB
    db.add_all(db_forecast_records)
    db.commit()

    # Format historical points (last 60 days for visualization)
    recent_history = hist_df.tail(60)
    historical_points = [
        {"date": row["date"].strftime("%Y-%m-%d"), "actual_demand": float(row["quantity"])}
        for _, row in recent_history.iterrows()
    ]

    current_stock = drug.inventory.current_stock if drug.inventory else 0
    stock_status = drug.inventory.calculate_status() if drug.inventory else "UNKNOWN"

    return schemas.ForecastResponse(
        drug_id=drug.drug_id,
        drug_name=drug.drug_name,
        category=drug.category,
        forecast_horizon_days=days_ahead,
        total_predicted_demand=round(total_forecasted, 2),
        average_daily_demand=round(avg_daily_demand, 2),
        daily_demand_std=round(daily_demand_std, 2),
        metrics=metrics,
        forecast_points=forecast_points,
        historical_points=historical_points,
        reorder_point_calculated=reorder_point,
        safety_stock_calculated=safety_stock,
        current_stock=current_stock,
        status=stock_status
    )
