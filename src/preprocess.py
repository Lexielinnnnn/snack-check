"""Preprocessing: train/val/test split and the sklearn pipeline (imputation + scaling + model)."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def split_data(df: pd.DataFrame, cfg: dict):
    """Stratified split into train / validation / test using ratios from config."""
    features = cfg["features"]
    target = cfg["target"]["name"]
    s = cfg["split"]

    X, y = df[features], df[target]

    # First carve out the test set, then split the rest into train and validation
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=s["test_size"], stratify=y, random_state=s["random_state"]
    )
    val_fraction = s["val_size"] / (1 - s["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=s["random_state"]
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(model_type: str, params: dict) -> Pipeline:
    """Median imputation (fiber is often missing) -> standard scaling -> classifier.
    Bundling everything in one Pipeline means the API can call predict on raw inputs."""
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type '{model_type}', choose from {list(MODELS)}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MODELS[model_type](**params)),
    ])
