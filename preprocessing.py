import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bootstrap import DATA_DIR

# 1.4 Million rows
CDS_AND_VINYL_JSON_PARAMS = {
    'file': 'CDs_and_Vinyl_5.json.gz',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}
# 1.1 Million rows
CELL_PHONE_JSON_PARAMS = {
    'file': 'Cell_Phones_and_Accessories_5.json.gz',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}
# 11.3 Million rows
CLOTHING_JSON_PARAMS = {
    'file': 'Clothing_Shoes_and_Jewelry_5.json',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}
# 6.7 Million rows
ELECTRONICS_JSON_PARAMS = {
    'file': 'Electronics_5.json.gz',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}
# 6.9 Million rows
HOME_AND_KITCHEN_JSON_PARAMS = {
    'file': 'Home_and_Kitchen_5.json.gz',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}
# 2.2 Million rows
KINDLE_STORE_JSON_PARAMS = {
    'file': 'Kindle_Store_5.json.gz',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}
# 3.4 Million rows
MOVIES_JSON_PARAMS = {
    'file': 'Movies_and_TV_5.json.gz',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}
# 2.8 Million rows
SPORTS_JSON_PARAMS = {
    'file': 'Sports_and_Outdoors_5.json.gz',
    'filetype': 'json',
    'features': "reviewText",
    'labels': "overall",
}

RANDOM_STATE = 42

def get_dataframe_file(params:dict, points:int = None, equalize: bool = False) -> pd.DataFrame:
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
    if equalize:
        df = equalize_num_examples_per_label(df, params)

    return df.filter([params['features'], params['labels']], axis=1)

# Equalize the number of samples per label
# Determine the number of unique labels, determine the number of entries per unique label, determine the label with
# the fewest entries. Drop entries from the other labels to equalize the number of entries per label
#
def equalize_num_examples_per_label(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    # Determine the labels
    labels = df[params['labels']]
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
        labels = df[params['labels']]
        indices = df.index[labels == label]
        df.drop(indices[:num_entries_per_label[label] - min_num_entries], inplace=True)
    return df

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