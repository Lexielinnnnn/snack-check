# 🥨 Snack Check — Is this snack ultra-processed?

**Live app:** https://snack-check.streamlit.app · **API:** https://snack-check-api-433582708621.us-central1.run.app ([interactive docs](https://snack-check-api-433582708621.us-central1.run.app/docs))

> If the app shows a "sleeping" screen, click **Wake up** (~30 s). The first prediction after idle can take ~10 s while Cloud Run starts a container.

A full-stack ML system that predicts whether a packaged food is **ultra-processed (NOVA group 4)** from
the nine numbers on its nutrition label. Data from Open Food Facts → cleaned and stored in Google Cloud
Storage → sklearn pipeline tracked with MLflow → FastAPI on Cloud Run → Streamlit app on Streamlit Community Cloud.

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
  Streamlit (Community Cloud) ──POST /predict──▶ FastAPI (Cloud Run)
```

## Cloud services used

| Service | Used for |
|---|---|
| Google Cloud Storage | Cleaned dataset and the trained `model.joblib` |
| Cloud Build | Builds the Docker image remotely |
| Artifact Registry | Stores the Docker image |
| Cloud Run | Hosts the public FastAPI `/predict` endpoint |
| Streamlit Community Cloud | Hosts the Streamlit front end |

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

**One-time project setup** (needs a project Owner). New GCP projects run Cloud Build as the Compute Engine
default service account, which can't read the build source by default:

```bash
PROJECT=<PROJECT_ID>
SA=<PROJECT_NUMBER>-compute@developer.gserviceaccount.com
for ROLE in roles/cloudbuild.builds.builder roles/storage.objectViewer roles/artifactregistry.writer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" --role="$ROLE" --condition=None
done
```

The person deploying needs **Editor + Cloud Run Admin + Project IAM Admin**. Editor alone can deploy, but
can't make the service public (`--allow-unauthenticated` silently fails and the URL returns 403).

**Deploy:**

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>      # the ID (e.g. project-ca4d...), not the project number
bash scripts/deploy_cloud_run.sh
```

The script enables the required APIs, creates the Artifact Registry repo if needed, downloads the
current model from GCS into the build context, builds the image with Cloud Build, deploys it to Cloud Run
with `--allow-unauthenticated`, and runs a smoke test against the public URL. Override settings inline,
e.g. `REGION=us-east1 SERVICE=snack-check-api bash scripts/deploy_cloud_run.sh`.

**Alternative without Cloud Build** (build locally, push, deploy). On Apple Silicon, `--platform linux/amd64`
is required because Cloud Run only runs amd64 images:

```bash
gcloud storage cp gs://snack-check-data-lexie/models/model.joblib models/model.joblib
gcloud auth configure-docker us-central1-docker.pkg.dev
IMAGE=us-central1-docker.pkg.dev/<PROJECT_ID>/snack-check/api:$(git rev-parse --short HEAD)
docker buildx build --platform linux/amd64 -t $IMAGE --push .
gcloud run deploy snack-check-api --image $IMAGE --region us-central1 --allow-unauthenticated --memory 1Gi --max-instances 2
```

To ship a new model, retrain (`python src/train.py` uploads it to GCS) and redeploy; each image tag is the
git commit, so every deployed revision maps to one model file.

Live API: https://snack-check-api-433582708621.us-central1.run.app

### Front end → Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with a GitHub account that is an **admin** of
   the repo (push access is not enough).
2. **Create app → Deploy a public app from GitHub**, main file `frontend/app.py`, branch `main`.
3. **Advanced settings:** Python 3.11; under **Secrets** add
   ```toml
   API_URL = "https://snack-check-api-433582708621.us-central1.run.app"
   ```
   Streamlit exposes top-level secrets as environment variables, which `app.py` reads.
4. Deploy. Dependencies come from `frontend/requirements.txt` (next to `app.py`).

Live app: https://snack-check.streamlit.app

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
