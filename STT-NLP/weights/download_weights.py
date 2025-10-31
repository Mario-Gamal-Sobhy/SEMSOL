#!/usr/bin/env python3
import os
import sys
import urllib.request
from pathlib import Path

# Configure your weight URLs here (replace with your links)
MODEL_URL = os.environ.get("STT_MODEL_URL", "")  # e.g. https://.../model.pt
PREPROC_URL = os.environ.get("STT_PREPROC_URL", "")  # e.g. https://.../preprocessor.pkl

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
MODEL_PATH = ART_DIR / "model_trainer" / "model.pt"
PREPROC_PATH = ART_DIR / "data_transformation" / "preprocessor.pkl"


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved: {dest}")


def main():
    if not MODEL_URL or not PREPROC_URL:
        print("Please set STT_MODEL_URL and STT_PREPROC_URL environment variables.")
        sys.exit(1)

    download(MODEL_URL, MODEL_PATH)
    download(PREPROC_URL, PREPROC_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
