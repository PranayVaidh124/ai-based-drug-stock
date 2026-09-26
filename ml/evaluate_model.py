"""
Model Evaluation Metrics (MAE, RMSE, R², MAPE)
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates regression model performance.
    Returns:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R2: Coefficient of Determination (R²)
    - MAPE: Mean Absolute Percentage Error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # Calculate MAPE safely without division by zero
    non_zero = y_true > 0
    if np.sum(non_zero) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    else:
        mape = 0.0

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mape": round(mape, 2)
    }

    print("\n" + "=" * 45)
    print("      DEMAND FORECASTING MODEL EVALUATION")
    print("=" * 45)
    print(f" Mean Absolute Error (MAE)  : {metrics['mae']:.3f} units")
    print(f" Root Mean Sq Error (RMSE)  : {metrics['rmse']:.3f} units")
    print(f" R² Determination Score     : {metrics['r2']:.4f}")
    print(f" Mean Abs Pct Error (MAPE)  : {metrics['mape']:.2f}%")
    print("=" * 45 + "\n")

    return metrics
