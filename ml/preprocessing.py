"""
Data Preprocessing, Cleaning, Outlier Handling, and Feature Engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


def clean_consumption_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw consumption records:
    - Drops duplicates and null rows
    - Converts date to datetime
    - Clips negative values and extreme anomalies (> 99.5th percentile per category)
    """
    df = df.copy()
    df.dropna(subset=["drug_id", "date", "quantity_consumed"], inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df["quantity_consumed"] = pd.to_numeric(df["quantity_consumed"], errors="coerce").fillna(0)
    df["quantity_consumed"] = df["quantity_consumed"].clip(lower=0)

    # Outlier clipping per drug: replace quantities > 4 * standard deviations above mean with the threshold
    cleaned_rows = []
    for drug_id, group in df.groupby("drug_id"):
        group = group.sort_values("date")
        mean_val = group["quantity_consumed"].mean()
        std_val = group["quantity_consumed"].std()
        if pd.notnull(std_val) and std_val > 0:
            upper_bound = mean_val + 4.0 * std_val
            group["quantity_consumed"] = group["quantity_consumed"].clip(upper=upper_bound)
        cleaned_rows.append(group)

    cleaned_df = pd.concat(cleaned_rows, ignore_index=True)
    cleaned_df.sort_values(["drug_id", "date"], inplace=True)
    return cleaned_df


def engineer_features(df: pd.DataFrame, category_map: Dict[str, int] = None) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """
    Engineers temporal and autoregressive features:
    - Calendar features: month, day, day_of_week, is_weekend, day_of_year
    - Autoregressive lag features: lag_1, lag_7, lag_14, lag_30
    - Rolling window statistics: rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_mean_30
    - Category encoding
    """
    df = clean_consumption_data(df)

    # Encode category
    if category_map is None:
        categories = sorted(df["category"].unique())
        category_map = {cat: idx for idx, cat in enumerate(categories)}

    df["category_code"] = df["category"].map(category_map).fillna(0).astype(int)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter

    # Grouped lag and rolling features per drug
    features_list = []
    for drug_id, group in df.groupby("drug_id"):
        group = group.sort_values("date").copy()

        # Lag features
        group["lag_1"] = group["quantity_consumed"].shift(1)
        group["lag_7"] = group["quantity_consumed"].shift(7)
        group["lag_14"] = group["quantity_consumed"].shift(14)
        group["lag_30"] = group["quantity_consumed"].shift(30)

        # Rolling statistics (using shifted series so target day is not included in window)
        shifted = group["quantity_consumed"].shift(1)
        group["rolling_mean_7"] = shifted.rolling(window=7, min_periods=1).mean()
        group["rolling_std_7"] = shifted.rolling(window=7, min_periods=1).std().fillna(0)
        group["rolling_mean_14"] = shifted.rolling(window=14, min_periods=1).mean()
        group["rolling_mean_30"] = shifted.rolling(window=30, min_periods=1).mean()

        features_list.append(group)

    featured_df = pd.concat(features_list, ignore_index=True)

    # Drop early initial rows where lag_30 is NaN to ensure high data quality
    featured_df.dropna(subset=["lag_1", "lag_7", "lag_14", "lag_30"], inplace=True)

    feature_cols = [
        "category_code",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "quarter",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_mean_30"
    ]

    return featured_df, category_map, feature_cols


def split_train_test(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "quantity_consumed",
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Performs a time-series chronological train/test split.
    Uses the latest test_ratio portion of historical dates for testing.
    """
    unique_dates = sorted(df["date"].unique())
    split_index = int(len(unique_dates) * (1 - test_ratio))
    split_date = unique_dates[split_index]

    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    train_df = df[train_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    return X_train, y_train, X_test, y_test, train_df, test_df
