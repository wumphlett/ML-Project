import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bootstrap import DATASET


RANDOM_STATE = 42


def get_dataframe(points: int = None) -> pd.DataFrame:
    if os.getenv("KAGGLE_dataset") == "kazanova/sentiment140":
        rows = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(DATASET, encoding="ISO-8859-1", names=rows)
    else:
        df = pd.read_csv(DATASET)
    if points and points < len(df):
        df = df.sample(points)
    return df.filter([os.getenv("DATASET_features"), os.getenv("DATASET_labels")], axis=1)


def get_subsets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1:].to_numpy()

    base_test_size = (int(os.getenv("TEST_split")) + int(os.getenv("VALIDATION_split"))) / 100
    validation_test_size = int(os.getenv("VALIDATION_split")) / (
        int(os.getenv("TEST_split")) + int(os.getenv("VALIDATION_split"))
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=base_test_size, random_state=RANDOM_STATE)
    X_test, X_validate, y_test, y_validate = train_test_split(
        X_test, y_test, test_size=validation_test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, X_validate, y_train.reshape(-1), y_test.reshape(-1), y_validate.reshape(-1)

def get_subsets_no_val(X, y):
    base_test_size = int(os.getenv("TEST_split")) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=base_test_size, random_state=RANDOM_STATE)
    return X_train, X_test, y_train.reshape(-1), y_test.reshape(-1)
