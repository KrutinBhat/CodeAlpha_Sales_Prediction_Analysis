"""
Data loading and validation utilities for the Sales Prediction project.
"""

from pathlib import Path
import pandas as pd


def load_data(filepath):
    """
    Load the advertising dataset from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def clean_column_names(df):
    """
    Standardize dataframe column names.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def validate_dataset(df, target_column="sales"):
    """
    Perform basic dataset validation.

    Returns
    -------
    dict
        Dataset quality summary.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    validation = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(
            exclude="number"
        ).columns.tolist(),
    }

    return validation


def remove_duplicates(df):
    """
    Remove duplicate observations.
    """
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing_values(df):
    """
    Handle missing values using median for numeric columns
    and mode for categorical columns.
    """
    df = df.copy()

    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isnull().any():
            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    return df