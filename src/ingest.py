"""
Ingest Open Food Facts data for snack-check.

Steps:
  1. Download food.parquet from Hugging Face (once, cached locally), then read it with DuckDB
  2. Flatten the nested `nutriments` list into one column per nutrient (per 100g)
  3. Clean: drop missing values and physically impossible values
  4. Reproducible random sample, saved as a local parquet file
  5. Upload to Google Cloud Storage

Usage:
  python src/ingest.py --limit 20000 --no-upload   # quick test on a small subset
  python src/ingest.py                             # full run + upload to GCS
"""

import argparse
from pathlib import Path

import duckdb
import yaml

# output column -> nutrient `name` in Open Food Facts
NUTRIENTS = {
    "energy_kcal": "energy-kcal",
    "energy_kj": "energy-kj",
    "fat": "fat",
    "saturated_fat": "saturated-fat",
    "carbohydrates": "carbohydrates",
    "sugars": "sugars",
    "fiber": "fiber",
    "proteins": "proteins",
    "salt": "salt",
    "sodium": "sodium",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_raw(cfg: dict) -> str:
    """Download the raw parquet from Hugging Face once; later runs reuse the local cache.
    Needs an HF token (`hf auth login` or the HF_TOKEN env var), otherwise you get rate-limited."""
    from huggingface_hub import hf_hub_download

    d = cfg["data"]
    print(f"Downloading {d['hf_repo_id']}/{d['hf_filename']} (several GB the first time; cached afterwards)...")
    return hf_hub_download(
        repo_id=d["hf_repo_id"],
        filename=d["hf_filename"],
        repo_type="dataset",
    )


def build_query(cfg: dict, raw_path: str, limit: int | None) -> str:
    d = cfg["data"]
    t = cfg["target"]
    limit_sql = f"LIMIT {int(limit)}" if limit else ""

    # Long format -> wide format: one column per nutrient, taking the per-100g value
    pivot_cols = ",\n        ".join(
        f"max(CASE WHEN n.name = '{off}' THEN struct_extract(n, '100g') END) AS {col}"
        for col, off in NUTRIENTS.items()
    )

    return f"""
WITH meta AS (
    SELECT DISTINCT ON (code)
        code,
        {t['source_column']} AS nova_group,
        additives_n,
        struct_extract(list_extract(product_name, 1), 'text') AS product_name,
        array_to_string(countries_tags, ',') AS countries,
        nutriments
    FROM read_parquet('{raw_path}')
    WHERE {t['source_column']} BETWEEN 1 AND 4
      AND additives_n IS NOT NULL
      AND coalesce(obsolete, false) = false
      AND len(nutriments) > 0
    {limit_sql}
),
long AS (
    SELECT code, unnest(nutriments) AS n FROM meta
),
wide AS (
    SELECT
        code,
        {pivot_cols}
    FROM long
    GROUP BY code
),
merged AS (
    SELECT
        m.code, m.product_name, m.countries, m.nova_group, m.additives_n,
        coalesce(w.energy_kcal, w.energy_kj / 4.184) AS energy_kcal,  -- fall back to kJ when kcal is missing
        w.fat, w.saturated_fat, w.carbohydrates, w.sugars, w.fiber, w.proteins,
        coalesce(w.salt, w.sodium * 2.5) AS salt                      -- fall back to sodium x 2.5 when salt is missing
    FROM meta m
    JOIN wide w USING (code)
),
filtered AS (
    SELECT
        *,
        CAST(nova_group = {t['positive_value']} AS INTEGER) AS {t['name']}
    FROM merged
    WHERE energy_kcal   BETWEEN 0 AND 900
      AND fat           BETWEEN 0 AND 100
      AND saturated_fat BETWEEN 0 AND 100
      AND carbohydrates BETWEEN 0 AND 100
      AND sugars        BETWEEN 0 AND 100
      AND proteins      BETWEEN 0 AND 100
      AND salt          BETWEEN 0 AND 100
      AND (fiber IS NULL OR fiber BETWEEN 0 AND 100)   -- fiber is often missing; imputed at training time
      AND additives_n   BETWEEN 0 AND 50
      AND saturated_fat <= fat + 0.5                   -- saturated fat cannot exceed total fat
      AND sugars        <= carbohydrates + 0.5         -- sugars cannot exceed carbohydrates
      AND fat + carbohydrates + proteins <= 105        -- macronutrients cannot exceed 100g
)
SELECT * FROM filtered
USING SAMPLE reservoir({int(d['sample_size'])} ROWS) REPEATABLE ({int(d['seed'])})
"""


def upload_to_gcs(local_path: Path, bucket: str, blob: str) -> None:
    from google.cloud import storage

    client = storage.Client()
    client.bucket(bucket).blob(blob).upload_from_filename(str(local_path))
    print(f"Uploaded to gs://{bucket}/{blob}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Open Food Facts data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only read the first N products, for quick tests")
    parser.add_argument("--no-upload", action="store_true",
                        help="Save locally only, do not upload to GCS")
    args = parser.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]

    out_dir = Path(d["local_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / d["clean_filename"]

    raw_path = download_raw(cfg)

    con = duckdb.connect()
    print("Reading and cleaning data...")
    df = con.sql(build_query(cfg, raw_path, args.limit)).df()
    df.to_parquet(out_path, index=False)

    target = cfg["target"]["name"]
    print(f"Saved {len(df):,} rows to {out_path}")
    print("Target distribution:")
    print(df[target].value_counts(normalize=True).round(3).to_string())
    print("Missing values:")
    print(df[cfg["features"]].isna().sum().to_string())

    if not args.no_upload:
        upload_to_gcs(out_path, d["gcs_bucket"], d["gcs_blob"])


if __name__ == "__main__":
    main()
