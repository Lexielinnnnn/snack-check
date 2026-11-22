"""
Download the cleaned dataset from Google Cloud Storage.

This is the entry point of the training pipeline: training always pulls data
from the cloud instead of relying on a local copy.

Usage:
  python src/download_data.py            # skip if the file already exists locally
  python src/download_data.py --force    # always re-download from GCS
"""

import argparse
from pathlib import Path

import yaml
from google.cloud import storage


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_data(cfg: dict, force: bool = False) -> Path:
    """Download the cleaned parquet from GCS and return its local path."""
    d = cfg["data"]
    local_path = Path(d["local_dir"]) / d["clean_filename"]

    if local_path.exists() and not force:
        print(f"Found local copy at {local_path}, skipping download (use --force to refresh)")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    blob = client.bucket(d["gcs_bucket"]).blob(d["gcs_blob"])
    blob.download_to_filename(str(local_path))
    print(f"Downloaded gs://{d['gcs_bucket']}/{d['gcs_blob']} -> {local_path}")
    return local_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cleaned data from GCS")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-download even if a local copy exists")
    args = parser.parse_args()

    download_data(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
