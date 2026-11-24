# snack-check: one image for ingestion, training, and serving
#   serve (default): FastAPI on $PORT (Cloud Run sets PORT=8080)
#   train          : runs src/train.py (needs GCS credentials, see README)
#   ingest         : runs src/ingest.py (needs Hugging Face and GCS credentials)
#
#   docker build -t snack-check .
#   docker run -p 8080:8080 snack-check                 # serve
#   docker run snack-check train --no-upload            # train (extra args go to train.py)
#   docker run snack-check ingest --no-upload           # ingest (extra args go to ingest.py)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY scripts/ scripts/
COPY docker-entrypoint.sh .
# Bake the model into the image if one is present locally (models/ may be empty;
# then the API downloads it from MODEL_GCS_URI at startup).
COPY models/ models/

RUN chmod +x docker-entrypoint.sh && useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["serve"]
