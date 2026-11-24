#!/usr/bin/env bash
# Build the image with Cloud Build, push it to Artifact Registry, deploy to Cloud Run (public).
#
# Prereqs (once): gcloud auth login && gcloud config set project <PROJECT_ID>
#                 + the IAM setup in README ("API -> Cloud Run"), otherwise Cloud Build
#                 fails with 403 and the service is not publicly reachable
# Usage:          bash scripts/deploy_cloud_run.sh
# Override any variable inline, e.g.  REGION=us-east1 bash scripts/deploy_cloud_run.sh
set -euo pipefail

# Never wait on hidden interactive prompts (some gcloud output below is silenced)
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-snack-check}"                  # Artifact Registry repository name
SERVICE="${SERVICE:-snack-check-api}"        # Cloud Run service name
MODEL_GCS_URI="${MODEL_GCS_URI:-gs://snack-check-data-lexie/models/model.joblib}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/api:${TAG}"

echo "Project: ${PROJECT_ID}  Region: ${REGION}  Image: ${IMAGE}"

# 1. Enable APIs (no-op if already enabled)
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 2. Create the Artifact Registry repo if it doesn't exist
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPO}" --repository-format=docker --location "${REGION}" \
    --description "snack-check images"

# 3. Bake the current model into the image (reproducible: image tag <-> model version)
mkdir -p models
gcloud storage cp "${MODEL_GCS_URI}" models/model.joblib

# 4. Build remotely with Cloud Build and push to Artifact Registry
gcloud builds submit --tag "${IMAGE}" .

# 5. Deploy, publicly accessible
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "MODEL_GCS_URI=${MODEL_GCS_URI}"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)')"
echo
echo "Deployed: ${URL}"
echo "Smoke test:"
curl -s "${URL}/health"; echo
curl -s -X POST "${URL}/predict" -H "Content-Type: application/json" \
  -d '{"energy_kcal": 520, "fat": 30, "saturated_fat": 12, "carbohydrates": 55, "sugars": 35, "fiber": 2, "proteins": 6, "salt": 0.4, "additives_n": 5}'
echo
