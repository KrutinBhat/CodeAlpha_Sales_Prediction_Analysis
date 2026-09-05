"""
Feature engineering utilities for the Sales Prediction project.
"""

import pandas as pd
import numpy as np


def create_advertising_features(df):
    """
    Create useful advertising-related features.

    Expected columns:
        tv
        radio
        newspaper
    """

    df = df.copy()

    required_columns = {"tv", "radio", "newspaper"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing advertising columns: {sorted(missing)}"
        )

    # Total advertising expenditure
    df["total_ad_spend"] = (
        df["tv"] +
        df["radio"] +
        df["newspaper"]
    )

    # Channel contribution percentages
    df["tv_share"] = (
        df["tv"] /
        df["total_ad_spend"].replace(0, np.nan)
    )

    df["radio_share"] = (
        df["radio"] /
        df["total_ad_spend"].replace(0, np.nan)
    )

    df["newspaper_share"] = (
        df["newspaper"] /
        df["total_ad_spend"].replace(0, np.nan)
    )

    # Interaction effects
    df["tv_radio_interaction"] = df["tv"] * df["radio"]
    df["tv_newspaper_interaction"] = df["tv"] * df["newspaper"]
    df["radio_newspaper_interaction"] = df["radio"] * df["newspaper"]

    # Non-linear effects
    df["tv_squared"] = df["tv"] ** 2
    df["radio_squared"] = df["radio"] ** 2
    df["newspaper_squared"] = df["newspaper"] ** 2

    # Avoid division by zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def get_feature_target_split(df, target_column="sales"):
    """
    Split dataframe into features and target.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y