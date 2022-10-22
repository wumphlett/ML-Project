import os
from pathlib import Path

from dotenv import load_dotenv
import kaggle


DATA_DIR = Path("./data").resolve()
DATASET = DATA_DIR / "dataset.csv"


def setup():
    boilerplate()


def boilerplate():
    load_dotenv(override=True)

    DATA_DIR.mkdir(exist_ok=True)
    download_data(os.getenv("KAGGLE_dataset"))


def download_data(dataset: str):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=DATA_DIR, unzip=True)

    for file in DATA_DIR.iterdir():
        if file.is_file() and "dataset" in file.stem.lower():
            file.rename(DATASET)
            break
