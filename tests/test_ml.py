"""
Machine Learning Pipeline Tests
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from ml.preprocessing import clean_consumption_data, engineer_features
from ml.evaluate_model import evaluate_predictions
from backend.services.forecasting_service import get_cached_model


def test_data_cleaning():
    raw_data = {
        "drug_id": [1, 1, 1, 2],
        "category": ["Antibiotics", "Antibiotics", "Antibiotics", "Cardiovascular"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01"],
        "quantity_consumed": [-5, 20, 500, 30]
    }
    df = pd.DataFrame(raw_data)
    cleaned = clean_consumption_data(df)

    # Negative value clipped to 0
    assert cleaned["quantity_consumed"].min() >= 0
    assert len(cleaned) == 4


def test_evaluation_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 19.0, 31.0, 38.0])

    metrics = evaluate_predictions(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics

    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["r2"] > 0.80


def test_loaded_model_bundle():
    bundle = get_cached_model()
    assert "model" in bundle
    assert "feature_cols" in bundle
    assert "category_map" in bundle
    assert hasattr(bundle["model"], "predict")
