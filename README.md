# 🥨 Snack Check — Is this snack ultra-processed?

**Live app:** <FRONTEND_URL> · **API:** <API_URL> ([interactive docs](<API_URL>/docs))

A full-stack ML system that predicts whether a packaged food is **ultra-processed (NOVA group 4)** from
the nine numbers on its nutrition label. Data from Open Food Facts → cleaned and stored in Google Cloud
Storage → sklearn pipeline tracked with MLflow → FastAPI on Cloud Run → Streamlit app on Hugging Face Spaces.

<!-- ===== Sections owned by A: overview & goals, dataset (Open Food Facts, ODbL), model & evaluation,
     how to train, ethics & limitations ===== -->

## Architecture

```
HF food.parquet ──ingest.py──▶ GCS (food_clean.parquet)
                                  │ download_data.py
                                  ▼
                 preprocess.py + train.py ──▶ MLflow (params, metrics, plots, model)
                                  │
                                  ▼
                     GCS models/model.joblib
                                  │ baked into image
                                  ▼
  Streamlit (HF Spaces) ──POST /predict──▶ FastAPI (Cloud Run)
```

## Cloud services used

| Service | Used for |
|---|---|
| Google Cloud Storage | Cleaned dataset and the trained `model.joblib` |
| Cloud Build | Builds the Docker image remotely |
| Artifact Registry | Stores the Docker image |
| Cloud Run | Hosts the public FastAPI `/predict` endpoint |
| Hugging Face Spaces | Hosts the Streamlit front end |

## API

`POST /predict`, all values per 100 g (`fiber` is optional; missing values are imputed by the model pipeline):

```json
{"energy_kcal": 520, "fat": 30, "saturated_fat": 12, "carbohydrates": 55,
 "sugars": 35, "fiber": 2, "proteins": 6, "salt": 0.4, "additives_n": 5}
```

Response:

```json
{"ultra_processed": 1, "probability": 0.87}
```

Input validation returns `422` for negative values, values above 100 g per 100 g (900 kcal for energy),
unknown fields, `saturated_fat > fat`, or `sugars > carbohydrates`. `GET /health` reports whether the
model is loaded and where it came from.

## Build and test the API locally

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Get a model: the real one from GCS ...
gcloud storage cp gs://snack-check-data-lexie/models/model.joblib models/model.joblib
# ... or a placeholder with the same interface (meaningless predictions)
python scripts/make_dummy_model.py

uvicorn src.api.main:app --reload --port 8080   # docs at http://localhost:8080/docs
pytest -q                                       # API tests
```

### With Docker

One image, two entry points (`serve` is the default):

```bash
docker build -t snack-check .

# Serve (uses models/model.joblib baked in at build time)
docker run --rm -p 8080:8080 snack-check

# Train (needs GCS credentials; mounts your gcloud login into the container)
gcloud auth application-default login
docker run --rm \
  -v ~/.config/gcloud:/home/appuser/.config/gcloud:ro \
  -e GOOGLE_CLOUD_PROJECT=<PROJECT_ID> \
  snack-check train --no-upload
```

If `models/` is empty at build time, set `-e MODEL_GCS_URI=gs://snack-check-data-lexie/models/model.joblib`
(plus the credentials mount) and the API downloads the model at startup.

### Test with curl

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"energy_kcal": 520, "fat": 30, "saturated_fat": 12, "carbohydrates": 55, "sugars": 35, "fiber": 2, "proteins": 6, "salt": 0.4, "additives_n": 5}'

# Invalid input -> 422
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"energy_kcal": -5, "fat": 30, "saturated_fat": 12, "carbohydrates": 55, "sugars": 35, "proteins": 6, "salt": 0.4, "additives_n": 5}'
```

## Deploy to the cloud

### API → Cloud Run

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
bash scripts/deploy_cloud_run.sh
```

The script enables the required APIs, creates the Artifact Registry repo if needed, downloads the
current model from GCS into the build context, builds the image with Cloud Build, deploys it to Cloud Run
with `--allow-unauthenticated`, and runs a smoke test against the public URL. Override settings inline,
e.g. `REGION=us-east1 SERVICE=snack-check-api bash scripts/deploy_cloud_run.sh`.

To ship a new model, retrain (`python src/train.py` uploads it to GCS) and rerun the deploy script; each
image tag is the git commit, so every deployed revision maps to one model file.

### Front end → Hugging Face Spaces

1. Create a new Space (SDK: **Docker**, blank template).
2. Copy the front-end files into the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/<HF_USER>/snack-check space
   cp frontend/app.py frontend/requirements.txt frontend/Dockerfile space/
   cp frontend/SPACE_README.md space/README.md
   cd space && git add . && git commit -m "Deploy Snack Check front end" && git push
   ```
3. In the Space **Settings → Variables and secrets**, add a variable `API_URL` = your Cloud Run URL.

Run the front end locally against a local API:

```bash
pip install -r frontend/requirements.txt
API_URL=http://localhost:8080 streamlit run frontend/app.py
```

## Front end

A form styled like the nutrition panel on a package (per 100 g + number of additives), with one-click
example snacks. It sends the values to the Cloud Run API with `requests` and shows the verdict, the model's
probability as a progress bar, and a short explanation of NOVA. The app checks the same consistency rules
as the API before sending.

**Front-end limitations:** the example snacks use approximate typical values, not specific products; users
have to count additives (E-numbers) themselves, which is error-prone; the first request after idle can take
~10 s while Cloud Run starts a container; and a single probability can read as more certain than the model
is, so the app labels it as model confidence and states that it isn't nutrition advice.
