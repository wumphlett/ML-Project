import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DIGITS = 5


def accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
    return accuracy_score(actual_labels, predicted_labels)


def confusion(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
    return confusion_matrix(actual_labels, predicted_labels)


def report(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> str:
    return classification_report(actual_labels, predicted_labels, digits=DIGITS)
