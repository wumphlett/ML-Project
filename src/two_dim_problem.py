from typing import Tuple

import matplotlib.pyplot as plt
from sympy import lambdify, sympify
import numpy as np


class TwoDimProblem:
    def __init__(self, value_range: int):
        self.seperator = None
        self.data = None
        self.value_range = value_range
        self.labels = None

    def create_data(self, soln_rank: int, noise_frac: float, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create data of length samples that is separable with a line of given rank with some noise"""
        assert 0 <= noise_frac < 1
        assert soln_rank > 0

        coefficients = np.random.randint(-self.value_range, self.value_range + 1, size=(soln_rank + 1))
        expression = "+".join([f"{c}*x**{i}" for i, c in enumerate(coefficients)])
        self.seperator = lambdify("x", sympify(expression))
        self.data = (np.random.rand(samples, 2) - 0.5) * self.value_range * 2
        self.labels = np.array([1 if (point[1] > self.seperator(point[0])) else 0 for point in self.data])

        noise_pts = int(samples * noise_frac)
        self.labels[:noise_pts] = np.mod(np.array(self.labels[:noise_pts]) + 1, 2)

        return self.data, self.labels

    def plot_data(self, show_seperator: bool = False):
        """Plot created data"""
        assert self.data is not None and self.labels is not None and self.seperator is not None
        RANGE = self.value_range
        plt.scatter(self.data[:, 0], self.data[:, 1], c=["b" if l == 1 else "orange" for l in self.labels])
        if show_seperator:
            x = np.linspace(-RANGE, RANGE, 1000)
            x = np.array([i for i in x if -RANGE * 1.5 <= self.seperator(i) <= RANGE * 1.5])
            y = self.seperator(x)
            plt.plot(x, y, c="black")
        plt.show()

    def plot_pred(self, y_pred, show_correct: bool = True, seperator=None):
        """Plot predicted data"""
        assert self.labels is not None and self.data is not None and self.seperator is not None
        assert len(y_pred) == len(self.labels)

        RANGE = self.value_range
        print(f"Accuracy = {np.mean(y_pred == self.labels)}")

        TP = np.logical_and(y_pred == 1, self.labels == 1)
        FP = np.logical_and(y_pred == 1, self.labels == 0)
        TN = np.logical_and(y_pred == 0, self.labels == 0)
        FN = np.logical_and(y_pred == 0, self.labels == 1)

        plt.scatter(self.data[TP, 0], self.data[TP, 1], c="g", label="TP")
        plt.scatter(self.data[FP, 0], self.data[FP, 1], c="r", label="FP")
        plt.scatter(self.data[TN, 0], self.data[TN, 1], c="lightgreen", label="TN")
        plt.scatter(self.data[FN, 0], self.data[FN, 1], c="#990000", label="FN")

        if seperator is not None:
            x = np.linspace(-RANGE, RANGE, 1000)
            x = np.array([i for i in x if -RANGE * 1.5 <= seperator(i) <= RANGE * 1.5])
            y = seperator(x)
            plt.plot(x, y, c="black")
        if show_correct:
            x = np.array(
                [i for i in np.linspace(-RANGE, RANGE, 1000) if -RANGE * 1.5 <= self.seperator(i) <= RANGE * 1.5]
            )
            y = self.seperator(x)
            plt.plot(x, y, c="green", linestyle="dashed")
        plt.legend()
        plt.title("Prediction Results for Model")
        plt.show()
