import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Sequence
from sklearn.preprocessing import OneHotEncoder
from numba import jit, njit, guvectorize, float64
from numba.typed import List
from compiled_functions import batch_backprop_compiled, batch_backprop

# @guvectorize([(float64[:], float64[:])], "(n)->(n)")
@njit
def sigmoid(X: np.ndarray):
    return 1.0 / (1.0 + np.exp(-X))
@njit
def dSigmoid(X: np.ndarray):
    a = 1.0 / (1.0 + np.exp(-X))
    return a*(1-a)
@njit
def tanh(X: np.ndarray):
    return np.tanh(X)
@njit
def dTanh(X: np.ndarray):
    return 1.0 - np.tanh(X)**2

class MultiLayerPerceptron:
    def __init__(
        self,
        epochs: int,
        lr: float,
        input_layer: int,
        output_layer: int,
        hidden_layers: Sequence[int],
        activation: str = "sigmoid",
    ):
        self.num_epochs = epochs
        self.lr = lr
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self._structure = (input_layer, *hidden_layers, output_layer)
        self._num_layers = len(self._structure)
        self._yencoder = None

        match activation:
            case "sigmoid" | "logistic":
                self.activation = sigmoid
                self.dActivation = dSigmoid
            case "tanh":
                self.activation = tanh
                self.dActivation = dTanh
            case _:
                raise ValueError("Invalid activation function")

        self._biases = [np.zeros((y, 1)) for y in self._structure[1:]]
        self._weights = [np.zeros((x,y)) for x, y in zip(self._structure[:-1], self._structure[1:])]
        self.loss_curve = []
    
    def epochs(self):
        for i in range(self.num_epochs):
            yield i, self.lr
    
    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1, compiled=True) -> None:
        self._biases = [np.random.randn(y, 1) for y in self._structure[1:]]
        self._weights = [np.random.randn(x,y) for x, y in zip(self._structure[:-1], self._structure[1:])]

        y = y[:, np.newaxis]
        if self.output_layer > 1:
            self._yencoder = OneHotEncoder(sparse=False)
            y = self._yencoder.fit_transform(y)

        for epoch_num, lr in tqdm(self.epochs(), total=self.num_epochs):
            loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                # print(f"X_batch: {X_batch.shape}")
                # print(f"y_batch: {y_batch.shape}")
                dJdB, dJdW = self._batch_backprop(X_batch, y_batch, compiled=compiled)
                loss += self._calc_loss(X_batch, y_batch)
                self._biases = [b - lr * db for b, db in zip(self._biases, dJdB)]
                self._weights = [w - lr * dw for w, dw in zip(self._weights, dJdW)]
            self.loss_curve.append(loss)
        
    def predict(self, X_list: np.ndarray) -> np.ndarray:
        pred = []
        for X in X_list:
            curr = X[:, np.newaxis]
            for W, b in zip(self._weights[:-1], self._biases[:-1]):
                z = W.T @ curr + b
                curr = self.activation(z)
            W, b = self._weights[-1], self._biases[-1]
            z = W.T @ curr + b
            if self.output_layer == 1:
                actual = 1 if z > 0.5 else 0
            else:
                actual = np.argmax(self.activation(z))
                
            pred.append(actual)
        return np.array(pred)
        

    def _batch_backprop(self, 
        X_batch: np.ndarray,
        y_batch: np.ndarray, 
        compiled = True) -> tuple[list[np.ndarray], list[np.ndarray]]:

        X_batch = X_batch[:,:,np.newaxis]
        y_batch = y_batch[:,:,np.newaxis]
        if compiled:
            weights = List(self._weights)
            biases = List(self._biases)
            return batch_backprop_compiled(X_batch, y_batch, weights, biases, self.activation, self.dActivation, self._structure)
        else:
            return batch_backprop(X_batch, y_batch, self._weights, self._biases, self.activation, self.dActivation, self._structure)
        
    def _calc_loss(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        loss = 0.0
        for X, y in zip(X_batch, y_batch):
            curr = X[:, np.newaxis]

            if self.output_layer != 1:
                y = y[:, np.newaxis]

            for W, b in zip(self._weights, self._biases):
                z = W.T @ curr + b
                # print(z)
                curr = sigmoid(z)
            # print((curr - y).shape)
            loss += np.sum((curr - y) ** 2)
        return loss
    
    def plot_loss(self) -> None:
        plt.plot(self.loss_curve)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.show()
