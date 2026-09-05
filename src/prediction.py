"""
Prediction utilities for the Sales Prediction project.
"""

from pathlib import Path

import joblib
import pandas as pd


def save_model(model, filepath):
    """
    Save a trained model using joblib.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, filepath)

    return filepath


def load_model(filepath):
    """
    Load a trained joblib model.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model not found: {filepath}"
        )

    return joblib.load(filepath)


def predict_sales(model, input_data):
    """
    Generate sales predictions.

    Parameters
    ----------
    model :
        Trained regression model.

    input_data : pandas.DataFrame
        Must contain the original training feature names:
        TV, Radio, Newspaper.

    Returns
    -------
    numpy.ndarray
        Predicted sales values.
    """

    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    required_columns = [
        "TV",
        "Radio",
        "Newspaper"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}. "
            f"Expected columns: {required_columns}"
        )

    input_data = input_data[required_columns]

    return model.predict(input_data)


def predict_single(model, tv, radio, newspaper):
    """
    Predict sales for a single advertising scenario.

    Parameters
    ----------
    tv : float
        TV advertising expenditure.

    radio : float
        Radio advertising expenditure.

    newspaper : float
        Newspaper advertising expenditure.
    """

    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(input_data)

    return float(prediction[0])