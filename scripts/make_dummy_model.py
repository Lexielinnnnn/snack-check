"""
Create a placeholder model.joblib so the API / Docker / Cloud Run setup can be tested
before (or without) the real model from src/train.py.

It has the same interface as the real model: an sklearn Pipeline taking the 9 features
in config order and exposing predict / predict_proba. Its predictions are meaningless.

Usage: python scripts/make_dummy_model.py [--out models/model.joblib]
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["energy_kcal", "fat", "saturated_fat", "carbohydrates",
            "sugars", "fiber", "proteins", "salt", "additives_n"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({
        "energy_kcal": rng.uniform(0, 600, n), "fat": rng.uniform(0, 40, n),
        "saturated_fat": rng.uniform(0, 15, n), "carbohydrates": rng.uniform(0, 80, n),
        "sugars": rng.uniform(0, 50, n), "fiber": rng.uniform(0, 10, n),
        "proteins": rng.uniform(0, 30, n), "salt": rng.uniform(0, 3, n),
        "additives_n": rng.integers(0, 12, n).astype(float),
    })[FEATURES]
    # Toy rule: more additives and sugar -> "ultra-processed"
    y = ((X["additives_n"] >= 3) | (X["sugars"] > 30)).astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"Saved DUMMY model to {out}")


if __name__ == "__main__":
    main()
