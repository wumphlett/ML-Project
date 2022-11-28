import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bootstrap import DATASET, DATA_DIR


RANDOM_STATE = 42

def get_dataframe_file(params:dict, points:int = None) -> pd.DataFrame:
    '''Get a pandas dataframe from a file
    params: dict containing 'file', 'encoding', 'rows', 'features', 'labels'
    points: number of points to return
    returns: pandas dataframe with features and labels
    '''
    validate_params(params)
    if params['filetype'] == 'csv':
        df = pd.read_csv( DATA_DIR / params['file'], encoding=params['encoding'], names=params['rows'])
    elif params['filetype'] == 'json':
        df = pd.read_json( DATA_DIR / params['file'], encoding=params['encoding'], lines=True)

    if points:
        df = df.sample(n=points, random_state=RANDOM_STATE)

    return df.filter([params['features'], params['labels']], axis=1)

def validate_params(params: dict):
    if 'file' not in params:
        raise Exception("params must contain 'file' key")
    if 'filetype' not in params:
        params['filetype'] = params['file'].split('.')[-1]
        if params['filetype'] not in ['csv', 'json']:
            raise Exception("params must contain 'filetype' key")
    if 'encoding' not in params:
        params['encoding'] = None
    if 'rows' not in params:
        params['rows'] = None
    if 'features' not in params:
        raise Exception("params must contain 'features' key")
    if 'labels' not in params:
        raise Exception("params must contain 'labels' key")
    
# def get_dataframe(points: int = None) -> pd.DataFrame:
#     if os.getenv("KAGGLE_dataset") == "kazanova/sentiment140":
#         rows = ["target", "ids", "date", "flag", "user", "text"]
#         df = pd.read_csv(DATASET, encoding="ISO-8859-1", names=rows)
#     else:
#         df = pd.read_csv(DATASET)
#     if points and points < len(df):
#         df = df.sample(points)
#     return df.filter([os.getenv("DATASET_features"), os.getenv("DATASET_labels")], axis=1)


def get_subsets(
    X:np.ndarray, 
    y:np.ndarray,
    train_split:float = 0.8,
    val_split:float = 0.1,
    test_split:float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns train, val, test subsets of the dataset
    X: features in a numpy array
    y: labels in a numpy array
    train_split: percentage of data to use for training
    val_split: percentage of data to use for validation
    test_split: percentage of data to use for testing
    returns: train_features, test_features, val_features, train_labels, test_labels, val_labels
    '''

    if train_split + val_split + test_split != 1:
        raise Exception("train_split + val_split + test_split must equal 1")

    base_test_size = 1 - train_split
    validation_test_size = val_split / (val_split + test_split)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=base_test_size, random_state=RANDOM_STATE)
    X_test, X_validate, y_test, y_validate = train_test_split(
        X_test, y_test, test_size=validation_test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, X_validate, y_train.reshape(-1), y_test.reshape(-1), y_validate.reshape(-1)