from datetime import datetime
from inspect import signature
from itertools import product
from multiprocessing import cpu_count, Pool
from typing import Any, Dict, Iterable, Union

import numpy as np
from sklearn.metrics import accuracy_score

from environment import DATA_DIR


def matrix_train(
    hyperparameters: Dict[str, Union[Iterable[Any], Any]],
    mlp,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    global train

    parameters = signature(mlp.__init__).parameters
    for parameter in hyperparameters:
        if parameter not in parameters:
            raise Exception(f"parameter {parameter} not found in the MLP init method")

    for parameter in parameters:
        if parameter not in hyperparameters and parameter != "self":
            raise Exception(f"parameter {parameter} not found in the training matrix")

    constants, axises = [], []
    for parameter, specified in hyperparameters.items():
        if isinstance(specified, Iterable):
            axises.append((parameter, specified))
        else:
            constants.append((parameter, specified))

    matrix = []
    for parameters in product(*[axis[1] for axis in axises]):
        matrix.append({name: parameter for name, parameter in zip([axis[0] for axis in axises], parameters)})

    for item in matrix:
        item.update({name: parameter for name, parameter in constants})

    combinations = len(matrix)
    digits = len(str(len(matrix)))
    logging_file = open(DATA_DIR / "train.log", "w")

    def train(parameter_idx):
        parameter_kwargs = matrix[parameter_idx]
        logging_file.write(
            f"{datetime.now().strftime('%X')} START {str(parameter_idx).ljust(digits)} / {combinations} : {parameter_kwargs}\n"
        )
        logging_file.flush()
        model = mlp(**parameter_kwargs)
        model.fit(X_train, y_train)
        logging_file.write(
            f"{datetime.now().strftime('%X')} ENDED {str(parameter_idx).ljust(digits)} / {combinations} : {parameter_kwargs}\n"
        )
        logging_file.flush()
        return accuracy_score(y_test, model.predict(X_test))

    with Pool(processes=cpu_count() - 1) as p:
        accuracies = p.map(train, range(len(matrix)))

    logging_file.close()

    with open(DATA_DIR / "train.csv", "w") as train_results:
        train_results.write(f"accuracy,kwargs\n")
        for accuracy, kwargs in zip(accuracies, matrix):
            train_results.write(f'{accuracy},"{kwargs}"\n')

    del train
    return matrix[max((idx for idx in range(len(matrix))), key=lambda x: accuracies[x])]
