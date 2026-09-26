"""
Random Forest Demand Forecasting Model Training Pipeline
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

from ml.preprocessing import engineer_features, split_train_test
from ml.evaluate_model import evaluate_predictions


def train_forecasting_model(
    csv_path: str = "data/drug_inventory.csv",
    model_output_path: str = "ml/model.pkl",
    n_estimators: int = 120,
    max_depth: int = 16,
    random_state: int = 42
):
    """
    Trains a Random Forest Regressor to forecast drug demand.
    Saves the model bundle (model, feature columns, category map, evaluation metrics)
    to a serialized joblib pickle file.
    """
    print(f"Loading consumption data from: {csv_path}")
    if not os.path.exists(csv_path):
        from data.generate_dataset import generate_full_dataset
        print("Data file not found. Generating realistic synthetic dataset first...")
        generate_full_dataset(output_csv=csv_path)

    df_raw = pd.read_csv(csv_path)
    print(f"Raw records loaded: {len(df_raw)}")

    # Feature engineering
    print("Performing feature engineering and temporal lag transformations...")
    featured_df, category_map, feature_cols = engineer_features(df_raw)
    print(f"Dataset after feature extraction: {len(featured_df)} rows, {len(feature_cols)} features.")

    # Train-test split (chronological)
    print("Splitting data chronologically into 80% train and 20% test sets...")
    X_train, y_train, X_test, y_test, train_df, test_df = split_train_test(
        featured_df, feature_cols, target_col="quantity_consumed", test_ratio=0.2
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train Random Forest Regressor
    print(f"Training Random Forest Regressor (n_estimators={n_estimators}, max_depth={max_depth})...")
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    # Evaluate model
    print("Evaluating model predictions on unseen test split...")
    y_pred = rf_model.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)  # Demand cannot be negative
    metrics = evaluate_predictions(y_test, y_pred)

    # Feature importances
    importances = dict(zip(feature_cols, rf_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top Predictive Features:")
    for feat, imp in top_features:
        print(f"  - {feat:18s}: {imp * 100:.2f}%")

    # Serialize trained bundle
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": rf_model,
        "feature_cols": feature_cols,
        "category_map": category_map,
        "metrics": metrics,
        "importances": importances
    }
    joblib.dump(bundle, model_output_path)
    print(f"Model bundle successfully saved to: {model_output_path}")

    return bundle


if __name__ == "__main__":
    train_forecasting_model()
