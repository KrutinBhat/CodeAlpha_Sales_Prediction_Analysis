"""
Machine learning model training and evaluation utilities.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Split the dataset into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a regression model using standard metrics.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def build_models(random_state=42):
    """
    Create a collection of regression models.

    Returns
    -------
    dict
        Dictionary containing regression models.
    """

    models = {

        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01))
        ]),

        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=10000
            ))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            loss="huber",
            random_state=random_state
        ),

        "HistGradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            random_state=random_state
        )
    }

    return models


def compare_models(X_train, X_test, y_train, y_test):
    """
    Train and compare multiple regression models.
    """

    models = build_models()

    results = []
    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({
            "Model": name,
            **metrics
        })

        trained_models[name] = model

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    return results_df, trained_models


def tune_gradient_boosting(X_train, y_train):
    """
    Hyperparameter optimization for Gradient Boosting Regression.
    """

    model = GradientBoostingRegressor(
        random_state=42
    )

    parameter_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search