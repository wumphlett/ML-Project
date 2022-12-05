import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DIGITS = 5


def accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
    """Accuracy score"""
    return accuracy_score(actual_labels, predicted_labels)


def confusion(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
    """Confusion matrix"""
    return confusion_matrix(actual_labels, predicted_labels)


def report(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> str:
    """Classification report"""
    return classification_report(actual_labels, predicted_labels, digits=DIGITS)


def plots(train_acc, test_acc, val_acc, train_loss=None, val_loss=None) -> None:
    """Accuracy score per dataset and loss plots"""
    if train_loss:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(train_loss, label="Training Loss")
        if val_loss:
            ax1.plot(val_loss, label="Validation Loss")
        ax1.set_title("Loss over epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.bar(
            [f"Train: {train_acc:.3f}", f"Val: {val_acc:.3f}", f"Test: {test_acc:.3f}"], [train_acc, val_acc, test_acc]
        )
        ax2.set_title("Accuracy on different sets")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)

        fig.tight_layout()
    else:
        plt.bar(
            [f"Train: {train_acc:.3f}", f"Val: {val_acc:.3f}", f"Test: {test_acc:.3f}"], [train_acc, val_acc, test_acc]
        )
        plt.title("Accuracy on different sets")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.show()
