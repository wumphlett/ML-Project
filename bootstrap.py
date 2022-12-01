import os
from pathlib import Path

from dotenv import load_dotenv


DATA_DIR = Path("./data").resolve()
DATASET = DATA_DIR / "dataset.csv"


def setup(download=True):
    boilerplate()
    if download:
        download_data_kaggle(os.getenv("KAGGLE_dataset"))
    validate_env()

def validate_env():
    weight_keys = ["TRAIN_split", "TEST_split", "VALIDATION_split"]
    weight_sum = sum(int(os.getenv(key)) for key in weight_keys)
    if weight_sum != 100:
        raise Exception(f"env values {' '.join(weight_keys)} must sum to 100 (currently {weight_sum})")


def boilerplate():
    load_dotenv(override=True)
    DATA_DIR.mkdir(exist_ok=True)

def download_data_file(url: str, filename: str, overwrite: bool = False):
    DATA_DIR.mkdir(exist_ok=True)
    import urllib.request

    new_path = DATA_DIR / filename
    if new_path.exists() and not overwrite:
        print(f"File {new_path} already exists, skipping download")
        return new_path
    print(f"Downloading {url + filename} to {new_path}")
    urllib.request.urlretrieve(url+filename, new_path)
    return new_path

def download_data_kaggle(dataset: str):
    import kaggle  # Kaggle must be imported after envvars are loaded with load_dotenv

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=DATA_DIR, unzip=True)

    for file in DATA_DIR.iterdir():
        if file.is_file() and "dataset" in file.stem.lower():
            file.rename(DATASET)
            break
