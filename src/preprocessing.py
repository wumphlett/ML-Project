from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .bootstrap import DATA_DIR


RANDOM_STATE = 42


def get_dataframe(files: List[dict], **kwargs) -> pd.DataFrame:
    """Get dataframe from datafiles"""
    frames = []
    for file in files:
        frames.append(get_dataframe_file(file, **kwargs))
    return pd.concat(frames)


def get_dataframe_file(params: dict, points: int = None, equalize: bool = False) -> pd.DataFrame:
    """Get dataframe from single file"""
    validate_params(params)

    if params["filetype"] == "csv":
        df = pd.read_csv(DATA_DIR / params["file"], encoding=params["encoding"], names=params["rows"])
    elif params["filetype"] == "json":
        df = pd.read_json(DATA_DIR / params["file"], encoding=params["encoding"], lines=True)
    else:
        print(f"Invalid filetype {params['filetype']}")
        return pd.DataFrame()

    if points:
        df = df.sample(n=points, random_state=RANDOM_STATE)
    if equalize:
        df = equalize_num_examples_per_label(df, params)

    return df.filter([params["features"], params["labels"]], axis=1).dropna()


def validate_params(params: dict):
    """Validate get_dataframe_file params"""
    if "file" not in params:
        raise Exception("params must contain 'file' key")
    if "filetype" not in params:
        params["filetype"] = params["file"].split(".")[-1]
        if params["filetype"] not in ["csv", "json"]:
            raise Exception("params must contain 'filetype' key")
    if "encoding" not in params:
        params["encoding"] = None
    if "rows" not in params:
        params["rows"] = None
    if "features" not in params:
        raise Exception("params must contain 'features' key")
    if "labels" not in params:
        raise Exception("params must contain 'labels' key")


def equalize_num_examples_per_label(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Equalize data examples by label"""
    # Determine the labels
    labels = df[params["labels"]]
    # Determine the unique labels
    unique_labels = set(labels)
    # Determine the number of entries per label
    num_entries_per_label = dict()
    for label in unique_labels:
        num_entries_per_label.update({label: len(df.index[labels == label])})
    # Determine the label with the fewest entries
    min_num_entries = min(num_entries_per_label.values())
    # Drop entries from each label to equalize the number of entries per label
    for label in unique_labels:
        labels = df[params["labels"]]
        indices = df.index[labels == label]
        df.drop(indices[: num_entries_per_label[label] - min_num_entries], inplace=True)
    return df


def get_subsets(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train, test, and validation sets"""
    if train_split + val_split + test_split != 1:
        raise Exception("train_split + val_split + test_split must equal 1")

    base_test_size = 1 - train_split
    validation_test_size = val_split / (val_split + test_split)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=base_test_size, random_state=RANDOM_STATE)
    X_test, X_validate, y_test, y_validate = train_test_split(
        X_test, y_test, test_size=validation_test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, X_validate, y_train.reshape(-1), y_test.reshape(-1), y_validate.reshape(-1)
