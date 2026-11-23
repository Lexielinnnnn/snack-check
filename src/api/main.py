"""
FastAPI service for the snack-check model.

Endpoints
  GET  /health   -> deployment status + whether the model is loaded
  POST /predict  -> {"ultra_processed": 0/1, "probability": float}

Model loading (first match wins)
  1. MODEL_PATH (default: models/model.joblib) if the file exists — e.g. baked into the image
  2. MODEL_GCS_URI (e.g. gs://snack-check-data-lexie/models/model.joblib) — downloaded at startup

Run locally:
  uvicorn src.api.main:app --reload --port 8080
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Feature order must match config.yaml `features` and the model A trained.
FEATURES = [
    "energy_kcal", "fat", "saturated_fat", "carbohydrates",
    "sugars", "fiber", "proteins", "salt", "additives_n",
]

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MODEL_GCS_URI = os.getenv("MODEL_GCS_URI", "")

state: dict = {"model": None, "source": None}


def download_from_gcs(uri: str, dest: Path) -> None:
    from google.cloud import storage

    if not uri.startswith("gs://"):
        raise ValueError(f"MODEL_GCS_URI must start with gs://, got {uri!r}")
    bucket, blob = uri[len("gs://"):].split("/", 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket).blob(blob).download_to_filename(str(dest))


def load_model():
    if not MODEL_PATH.exists() and MODEL_GCS_URI:
        print(f"Downloading model from {MODEL_GCS_URI} ...")
        download_from_gcs(MODEL_GCS_URI, MODEL_PATH)
        state["source"] = MODEL_GCS_URI
    elif MODEL_PATH.exists():
        state["source"] = str(MODEL_PATH)
    else:
        raise FileNotFoundError(
            f"No model at {MODEL_PATH} and MODEL_GCS_URI is not set. "
            "Run `python scripts/make_dummy_model.py` for a placeholder model."
        )
    state["model"] = joblib.load(MODEL_PATH)
    print(f"Loaded model from {state['source']}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Snack Check API",
    description="Predicts whether a food is ultra-processed (NOVA 4) from its nutrition label.",
    version="1.0.0",
    lifespan=lifespan,
)
# Lets a browser-based front end call the API directly (Streamlit calls server-side, so it's optional).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NutritionLabel(BaseModel):
    """Per-100g values from the back of the package. Field names = model features."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "energy_kcal": 520, "fat": 30, "saturated_fat": 12, "carbohydrates": 55,
            "sugars": 35, "fiber": 2, "proteins": 6, "salt": 0.4, "additives_n": 5,
        }},
    )

    energy_kcal: float = Field(..., ge=0, le=900, description="kcal per 100g")
    fat: float = Field(..., ge=0, le=100, description="g per 100g")
    saturated_fat: float = Field(..., ge=0, le=100, description="g per 100g")
    carbohydrates: float = Field(..., ge=0, le=100, description="g per 100g")
    sugars: float = Field(..., ge=0, le=100, description="g per 100g")
    fiber: float | None = Field(None, ge=0, le=100, description="g per 100g; optional (often missing on labels)")
    proteins: float = Field(..., ge=0, le=100, description="g per 100g")
    salt: float = Field(..., ge=0, le=100, description="g per 100g")
    additives_n: int = Field(..., ge=0, le=100, description="number of additives (E-numbers) in the ingredient list")


class Prediction(BaseModel):
    ultra_processed: int
    probability: float


@app.get("/")
def root():
    return {"service": "snack-check", "docs": "/docs", "predict": "POST /predict", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "model_source": state["source"]}


@app.post("/predict", response_model=Prediction)
def predict(label: NutritionLabel) -> Prediction:
    if label.saturated_fat > label.fat:
        raise HTTPException(422, "saturated_fat cannot be greater than fat")
    if label.sugars > label.carbohydrates:
        raise HTTPException(422, "sugars cannot be greater than carbohydrates")

    model = state["model"]
    if model is None:
        raise HTTPException(503, "Model not loaded")

    row = pd.DataFrame([label.model_dump()], columns=FEATURES).astype("float64")  # None -> NaN, imputer fills it
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])
    return Prediction(ultra_processed=pred, probability=round(proba, 4))
