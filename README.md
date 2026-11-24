# 🥨 Snack Check — Is this snack ultra-processed?

**Live app:** https://snack-check.streamlit.app · **API:** https://snack-check-api-433582708621.us-central1.run.app ([interactive docs](https://snack-check-api-433582708621.us-central1.run.app/docs))

> If the app shows a "sleeping" screen, click **Wake up** (~30 s). The first prediction after idle can take ~10 s while Cloud Run starts a container.

A full-stack ML system that predicts whether a packaged food is **ultra-processed (NOVA group 4)** from
the nine numbers on its nutrition label. Data from Open Food Facts → cleaned and stored in Google Cloud
Storage → sklearn pipeline tracked with MLflow → FastAPI on Cloud Run → Streamlit app on Streamlit Community Cloud.

## Project overview and goals

You often hear that "ultra-processed" food is bad for you, but a package doesn't tell you whether it is.
The NOVA system, which sorts foods into four processing levels, depends on the ingredients and how the food
was made. Shoppers rarely read that closely, but every package has a nutrition label. We wanted to see how
far the label numbers plus the additive count can get you in predicting whether a product is NOVA 4
(ultra-processed).

The course's focus was the engineering side, so most of our effort went into making the pipeline
reproducible: the data is pulled from cloud storage, every parameter lives in `config.yaml`, runs are
tracked in MLflow, and the model is served as a containerized API with a small public front end.

## Dataset

We use [Open Food Facts](https://world.openfoodfacts.org/), a crowdsourced, non-profit food product
database, via the Hugging Face dataset
[`openfoodfacts/product-database`](https://huggingface.co/datasets/openfoodfacts/product-database)
(`food.parquet`, about 7.7 GB). The data is licensed under the ODbL (see [Data attribution](#data-attribution)).

`src/ingest.py` turns the raw file into a clean training set:

1. Download `food.parquet` once and cache it locally. We originally read it straight from Hugging Face
   with DuckDB, but the many small anonymous requests got us rate-limited (HTTP 429). Now we download once,
   logged in with an HF token, and read the local copy.
2. Keep products that have a NOVA group (1–4) and an additive count.
3. Flatten the nested `nutriments` field into one column per nutrient, using per-100 g values only. Some
   products only have per-serving values; we drop those rather than guess the serving size. When kcal or
   salt is missing, we compute it from kJ or sodium.
4. Remove impossible rows, such as nutrients above 100 g per 100 g, more saturated fat than total fat, or
   more sugar than carbohydrates.
5. Take a random sample of 100,000 products (seed 42) and upload it to
   `gs://snack-check-data-lexie/data/food_clean.parquet`.

Training always starts by downloading that file from GCS (`src/download_data.py`).

The nine features are the per-100 g values of `energy_kcal`, `fat`, `saturated_fat`, `carbohydrates`,
`sugars`, `fiber`, `proteins` and `salt`, plus `additives_n` (number of additives). `fiber` is missing for
about 19% of products, so the model imputes it. The target `ultra_processed` is 1 for NOVA 4 and 0
otherwise; 66% of the cleaned sample is ultra-processed.

## Model and evaluation

The model is one scikit-learn pipeline: median imputation → standard scaling → random forest (200 trees,
`max_depth=12`, `min_samples_leaf=5`, balanced class weights). Because preprocessing is inside the
pipeline, the API can pass raw inputs straight to `predict`. The data is split 70/15/15 into
train/validation/test, stratified by the target. We report F1 and AUROC since the classes are imbalanced.

We tried logistic regression as a baseline:

| Model | Split | F1 | AUROC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| Logistic regression | validation | 0.844 | 0.903 | 0.812 | 0.939 | 0.766 |
| Logistic regression | test | 0.849 | 0.904 | 0.818 | 0.938 | 0.776 |
| Random forest | validation | 0.899 | 0.945 | 0.871 | 0.938 | 0.863 |
| Random forest | test | 0.899 | 0.947 | 0.872 | 0.936 | 0.865 |

The random forest is better on almost every metric, and its validation and test scores are nearly the
same, so we deployed it. On the 15,000 test products it got 13,075 right. It missed 1,343 ultra-processed
products and wrongly flagged 582 others. That imbalance makes sense: some ultra-processed foods have
ordinary nutrition numbers, and the clues are in the ingredient list, which the model never sees.

Each MLflow run logs the parameters, metrics, a confusion matrix, a feature-importance plot and the model.
One snag: newer MLflow versions run a safety check when saving sklearn models and rejected our random
forest's internal types, so `train.py` explicitly lists the two types we trust.

## How to train the model

All settings are in `config.yaml`. To run the pipeline locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login                              # free Hugging Face token, needed for the download
gcloud auth application-default login      # lets Python access the GCS bucket

python src/ingest.py                       # optional: rebuild the cleaned data and upload it to GCS
python src/train.py --no-upload --model-type logistic_regression   # baseline
python src/train.py                        # random forest; uploads models/model.joblib to GCS

mlflow ui --backend-store-uri sqlite:///mlflow.db                  # view runs at http://127.0.0.1:5000
```

`ingest.py --limit 20000 --no-upload` runs a quick test on a subset. The uploaded model at
`gs://snack-check-data-lexie/models/model.joblib` is the one the deploy script packages into the API. To
train inside Docker instead, see [With Docker](#with-docker).

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

## Ethics and limitations

The model predicts a label; it doesn't measure how processed a food is. NOVA is based on ingredients and
manufacturing, and we only give the model nutrition numbers and an additive count. It picks up on patterns
that go along with ultra-processing, like more additives, sugar and salt, and it misses products whose
processing only shows in the ingredient list. The result also isn't health advice. Butter and sugar are
not NOVA 4, and NOVA itself is debated among nutrition researchers. We don't think the model should be
used for product labeling or marketing claims.

The data has its own problems. Open Food Facts is filled in by volunteers, so some values are wrong in ways
our cleaning can't catch, and the NOVA groups are computed automatically from the entered ingredients. Most
products come from Europe (especially France) and North America, so the model is probably less reliable
for food from other regions. We also only kept products with complete per-100 g data, which may not
represent what's actually on shelves. And since two-thirds of our sample is ultra-processed, the model's
probabilities reflect that mix rather than a typical shopping cart.

On the user side, people have to count additives themselves, which is easy to get wrong and affects the
prediction a lot. The front end presents the probability as model confidence and notes that it isn't
nutrition advice. No personal data is collected. Using the ingredient text, or letting users scan a barcode
instead of typing numbers, would probably help the most.

## Data attribution

Product data © [Open Food Facts](https://world.openfoodfacts.org/) contributors, available under the
[Open Database License](https://opendatacommons.org/licenses/odbl/1-0/); individual contents under the
[Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/). The cleaned sample in our
bucket is a derivative of this database.
