"""API tests. Uses whatever model is at MODEL_PATH (a dummy model is created if none exists).

Run: pytest -q
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
if not MODEL.exists():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dummy_model.py"), "--out", str(MODEL)], check=True)
os.environ["MODEL_PATH"] = str(MODEL)

from src.api.main import app  # noqa: E402

SAMPLE = {"energy_kcal": 520, "fat": 30, "saturated_fat": 12, "carbohydrates": 55,
          "sugars": 35, "fiber": 2, "proteins": 6, "salt": 0.4, "additives_n": 5}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_contract(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ultra_processed", "probability"}
    assert body["ultra_processed"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_fiber_optional(client):
    r = client.post("/predict", json={k: v for k, v in SAMPLE.items() if k != "fiber"})
    assert r.status_code == 200


@pytest.mark.parametrize("bad", [
    {**SAMPLE, "sugars": -1},                       # negative
    {**SAMPLE, "fat": 150},                         # > 100 g per 100 g
    {**SAMPLE, "saturated_fat": 40},                # saturated fat > fat
    {**SAMPLE, "sugars": 60},                       # sugars > carbohydrates
    {k: v for k, v in SAMPLE.items() if k != "salt"},  # missing required field
    {**SAMPLE, "colour": "red"},                    # unknown field
])
def test_invalid_input_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422
