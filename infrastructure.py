import pandas as pd
from sklearn.model_selection import train_test_split

from environment import DATASET


RANDOM_STATE=42


def get_subsets():
    df = pd.read_csv(DATASET)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=RANDOM_STATE)
    X_test, X_validate, y_test, y_validate = train_test_split(
        X_test, y_test, test_size = 0.5, random_state=RANDOM_STATE
    )
    return X_train, X_test, X_validate, y_train, y_test, y_validate
