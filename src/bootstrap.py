from pathlib import Path
import urllib.request

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


__all__ = (
    "DATA_DIR",
    "CDS_AND_VINYL_JSON_PARAMS",
    "CELL_PHONE_JSON_PARAMS",
    "CLOTHING_JSON_PARAMS",
    "ELECTRONICS_JSON_PARAMS",
    "HOME_AND_KITCHEN_JSON_PARAMS",
    "KINDLE_STORE_JSON_PARAMS",
    "MOVIES_JSON_PARAMS",
    "SPORTS_JSON_PARAMS",
    "setup",
    "download_data_file",
)

URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/"

DATA_DIR = Path("./data").resolve()

# 1.4 Million rows
CDS_AND_VINYL_JSON_PARAMS = {
    "file": "CDs_and_Vinyl_5.json.gz",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}
# 1.1 Million rows
CELL_PHONE_JSON_PARAMS = {
    "file": "Cell_Phones_and_Accessories_5.json.gz",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}
# 11.3 Million rows
CLOTHING_JSON_PARAMS = {
    "file": "Clothing_Shoes_and_Jewelry_5.json",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}
# 6.7 Million rows
ELECTRONICS_JSON_PARAMS = {
    "file": "Electronics_5.json.gz",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}
# 6.9 Million rows
HOME_AND_KITCHEN_JSON_PARAMS = {
    "file": "Home_and_Kitchen_5.json.gz",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}
# 2.2 Million rows
KINDLE_STORE_JSON_PARAMS = {
    "file": "Kindle_Store_5.json.gz",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}
# 3.4 Million rows
MOVIES_JSON_PARAMS = {
    "file": "Movies_and_TV_5.json.gz",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}
# 2.8 Million rows
SPORTS_JSON_PARAMS = {
    "file": "Sports_and_Outdoors_5.json.gz",
    "filetype": "json",
    "features": "reviewText",
    "labels": "overall",
}


def setup():
    """Setup environment"""
    DATA_DIR.mkdir(exist_ok=True)


def download_data_file(filename: str, overwrite: bool = False):
    """Download datafiles"""
    new_path = DATA_DIR / filename

    if new_path.exists() and not overwrite:
        print(f"File {new_path} already exists, skipping download")
        return new_path
    print(f"Downloading {URL + filename} to {new_path}")
    urllib.request.urlretrieve(URL + filename, new_path)
    return new_path
