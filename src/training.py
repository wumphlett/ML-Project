from datetime import datetime
from inspect import signature
from itertools import product
from multiprocessing import cpu_count, Pool
from typing import Any, Dict, Iterable, Union

import numpy as np
from sklearn.metrics import accuracy_score

from .bootstrap import DATA_DIR


def matrix_train(
    hyperparameters: Dict[str, Iterable[Any]],
    mlp,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Train multiple MLPs given a hypermatrix of hyperparameters"""
    global train

    parameters = signature(mlp.__init__).parameters
    for parameter in hyperparameters:
        if parameter not in parameters:
            raise Exception(f"parameter {parameter} not found in the MLP init method")

    axises = list(hyperparameters.items())

    matrix = []
    for parameters in product(*[axis[1] for axis in axises]):
        matrix.append({name: parameter for name, parameter in zip([axis[0] for axis in axises], parameters)})

    combinations = len(matrix)
    digits = len(str(len(matrix)))
    logging_file = open(DATA_DIR / "train.log", "w")

    def train(parameter_idx):
        parameter_kwargs = matrix[parameter_idx]
        logging_file.write(
            f"{datetime.now().strftime('%X')} START {str(parameter_idx+1).ljust(digits)} / {combinations} : {parameter_kwargs}\n"
        )
        logging_file.flush()
        model = mlp(**parameter_kwargs)
        model.fit(X_train, y_train, batch_size=200, output=False)
        acc = accuracy_score(y_test, model.predict(X_test))
        logging_file.write(
            f"{datetime.now().strftime('%X')} ENDED {str(parameter_idx+1).ljust(digits)} / {combinations} : {parameter_kwargs} ({acc})\n"
        )
        logging_file.flush()
        return acc

    with Pool(processes=cpu_count() - 1) as p:
        accuracies = p.map(train, range(len(matrix)))

    logging_file.close()

    with open(DATA_DIR / "train.csv", "w") as train_results:
        train_results.write(f"accuracy,kwargs\n")
        for accuracy, kwargs in zip(accuracies, matrix):
            train_results.write(f'{accuracy},"{kwargs}"\n')

    del train
    return matrix[max((idx for idx in range(len(matrix))), key=lambda x: accuracies[x])]
