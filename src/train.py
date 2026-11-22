"""
Train the snack-check model and track the run with MLflow.

Pipeline: download data from GCS -> split -> train -> evaluate -> log to MLflow
          -> save model.joblib -> upload model to GCS

Usage:
  python src/train.py                                   # model type from config.yaml
  python src/train.py --model-type logistic_regression  # override model type for a quick comparison
  python src/train.py --no-upload                       # don't upload the model to GCS
"""

import argparse
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed (also works inside Docker)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from download_data import download_data
from preprocess import build_pipeline, split_data


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(pipe, X, y, prefix: str) -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_auroc": roc_auc_score(y, proba),
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred),
        f"{prefix}_recall": recall_score(y, pred),
    }


def plot_confusion_matrix(pipe, X, y, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X, y, display_labels=["not ultra-processed", "ultra-processed"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(pipe, features: list, path: Path) -> None:
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Importance"
    else:  # logistic regression: absolute value of standardized coefficients
        values, label = np.abs(model.coef_[0]), "|Coefficient| (standardized)"

    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(features)[order], values[order])
    ax.set_xlabel(label)
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def upload_model(local_path: Path, bucket: str, blob: str) -> None:
    from google.cloud import storage

    storage.Client().bucket(bucket).blob(blob).upload_from_filename(str(local_path))
    print(f"Uploaded model to gs://{bucket}/{blob}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the snack-check model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-type", default=None,
                        help="Override model.type in config (logistic_regression / random_forest)")
    parser.add_argument("--no-upload", action="store_true", help="Don't upload the model to GCS")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = args.model_type or cfg["model"]["type"]
    model_params = cfg["model"]["params"][model_type]
    features = cfg["features"]

    # 1. Data (always pulled from the cloud)
    data_path = download_data(cfg)
    df = pd.read_parquet(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)
    print(f"Train {len(X_train):,} / val {len(X_val):,} / test {len(X_test):,}")

    # 2. MLflow setup
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_type):
        mlflow.log_params({
            "model_type": model_type,
            **{f"model__{k}": v for k, v in model_params.items()},
            **{f"split__{k}": v for k, v in cfg["split"].items()},
            "n_rows": len(df),
            "n_features": len(features),
            "features": ",".join(features),
        })

        # 3. Train
        pipe = build_pipeline(model_type, model_params)
        pipe.fit(X_train, y_train)

        # 4. Evaluate
        metrics = {**evaluate(pipe, X_val, y_val, "val"), **evaluate(pipe, X_test, y_test, "test")}
        mlflow.log_metrics(metrics)
        for k, v in metrics.items():
            print(f"{k:16s} {v:.4f}")

        # 5. Plots as artifacts
        with tempfile.TemporaryDirectory() as tmp:
            cm_path = Path(tmp) / "confusion_matrix.png"
            fi_path = Path(tmp) / "feature_importance.png"
            plot_confusion_matrix(pipe, X_test, y_test, cm_path)
            plot_feature_importance(pipe, features, fi_path)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # 6. Save model: local joblib (for the API) + MLflow model
        out = cfg["output"]
        model_path = Path(out["model_dir"]) / out["model_filename"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="joblib")
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            input_example=X_train.head(3).astype("float64"),  # float schema so missing values are allowed
            # We trained this model ourselves in this run, so these sklearn/numpy types are safe to trust
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        print(f"Saved model to {model_path}")

        # 7. Upload model to GCS for the API to load
        if not args.no_upload:
            upload_model(model_path, cfg["data"]["gcs_bucket"], out["gcs_blob"])


if __name__ == "__main__":
    main()
