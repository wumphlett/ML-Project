import numpy as np
from numba import njit, jit
from numba.core.errors import NumbaPerformanceWarning
import warnings

# theres some weird thing with @ operator that numba doesn't like
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# @jit
def fit_loop_compiled(X, y, weights, biases, structure, activation, dActivation, lr, num_epochs, batch_size):
    losses = [0.0] * num_epochs
    for epoch in range(num_epochs):
        loss = 0.0
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            dJdB, dJdW = batch_backprop(X_batch, y_batch, weights, biases, activation, dActivation, structure)
            loss += calc_loss(X_batch, y_batch, weights, biases, activation)
            biases = [b - lr * db for b, db in zip(biases, dJdB)]
            weights = [w - lr * dw for w, dw in zip(weights, dJdW)]
        losses[epoch] = loss
    return (weights, biases, loss)

@njit
def backprop_compiled( 
        X: np.ndarray, 
        y: np.ndarray,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        activation: np.ufunc,
        dActivation: np.ufunc,
        structure: tuple) -> tuple[list[np.ndarray], list[np.ndarray]]:

        dJdB = [np.zeros(b.shape, dtype=np.float64) for b in biases]
        dJdW = [np.zeros(w.shape, dtype=np.float64) for w in weights]
        
        layer_raw = [np.zeros(b.shape) for b in biases]
        layer_activations = [np.zeros(b.shape) for b in biases]
        a = X
        for i, (b, W) in enumerate(zip(biases, weights)):
            z = W.T @ a + b 
            a = activation(z)
            layer_raw[i] = z
            layer_activations[i] = a
        # For last layer, compare to y
        H = len(structure) - 2
        delta = (layer_activations[H] - y) * dActivation(layer_raw[H])
        dJdB[H] = delta
        dJdW[H] = layer_activations[H-1] @ delta.T

        # For all hidden layers, compare to layer after it
        for L in range(H-1, 0, -1):
            delta = weights[L+1] @ delta * dActivation(layer_raw[L])
            dJdB[L] = delta
            dJdW[L] = layer_activations[L-1] @ delta.T

        # For input layer, update W according to input, not previous layer
        delta = (weights[1] @ delta) * dActivation(layer_raw[0])
        dJdB[0] = delta
        dJdW[0] = X @ delta.T
        return (dJdB, dJdW)

@njit
def batch_backprop_compiled(X_batch: np.ndarray, 
    y_batch: np.ndarray, 
    weights: list[np.ndarray], 
    biases: list[np.ndarray], 
    activation: np.ufunc, 
    dActivation: np.ufunc, 
    structure: tuple,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:

    dJdB = [np.zeros(b.shape, dtype=np.float64) for b in biases]
    dJdW = [np.zeros(w.shape, dtype=np.float64) for w in weights]
    batch_size = X_batch.shape[0]
    for X, y in zip(X_batch, y_batch):
        dB, dW = backprop_compiled(X, y, weights, biases, activation, dActivation, structure)
        dJdB = [db + ddb for db, ddb in zip(dJdB, dB)]
        dJdW = [dw + ddw for dw, ddw in zip(dJdW, dW)]
    dJdB = [db / batch_size for db in dJdB]
    dJdW = [dw / batch_size for dw in dJdW]
    return (dJdB, dJdW)

def fit_loop(X, y, weights, biases, structure, activation, dActivation, lr, num_epochs, batch_size):
    losses = [0.0] * num_epochs
    for epoch in range(num_epochs):
        loss = 0.0
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            dJdB, dJdW = batch_backprop(X_batch, y_batch, weights, biases, activation, dActivation, structure)
            loss += calc_loss(X_batch, y_batch, weights, biases, activation)
            biases = [b - lr * db for b, db in zip(biases, dJdB)]
            weights = [w - lr * dw for w, dw in zip(weights, dJdW)]
        losses[epoch] = loss
    return (weights, biases, loss)

def batch_backprop(X_batch: np.ndarray, 
    y_batch: np.ndarray, 
    weights: list[np.ndarray], 
    biases: list[np.ndarray], 
    activation: np.ufunc, 
    dActivation: np.ufunc, 
    structure: tuple,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:

    dJdB = [np.zeros(b.shape) for b in biases]
    dJdW = [np.zeros(w.shape) for w in weights]
    batch_size = X_batch.shape[0]
    for X, y in zip(X_batch, y_batch):
        print(f"X: {X.shape}, y: {y.shape}")
        dB, dW = backprop(X, y, weights, biases, activation, dActivation, structure)
        dJdB = [db + ddb for db, ddb in zip(dJdB, dB)]
        dJdW = [dw + ddw for dw, ddw in zip(dJdW, dW)]
    dJdB = [db / batch_size for db in dJdB]
    dJdW = [dw / batch_size for dw in dJdW]
    return (dJdB, dJdW)


def backprop( 
        X: np.ndarray, 
        y: np.ndarray,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        activation: np.ufunc,
        dActivation: np.ufunc,
        structure: tuple) -> tuple[list[np.ndarray], list[np.ndarray]]:


        dJdB = [np.zeros(b.shape) for b in biases]
        dJdW = [np.zeros(w.shape) for w in weights]
        
        layer_raw = [np.zeros(b.shape) for b in biases]
        layer_activations = [np.zeros(b.shape) for b in biases]
        a = X
        for i, (b, W) in enumerate(zip(biases, weights)):
            z = W.T @ a + b 
            a = activation(z)
            layer_raw[i] = z
            layer_activations[i] = a

        # For last layer, compare to y
        H = len(structure) - 2
        delta = (layer_activations[H] - y) * dActivation(layer_raw[H])
        dJdB[H] = delta
        dJdW[H] = layer_activations[H-1] @ delta.T

        # For all hidden layers, compare to layer after it
        for L in range(H-1, 0, -1):
            delta = weights[L+1] @ delta * dActivation(layer_raw[L])
            dJdB[L] = delta
            dJdW[L] = layer_activations[L-1] @ delta.T

        # For input layer, update W according to input, not previous layer
        delta = (weights[1] @ delta) * dActivation(layer_raw[0])
        dJdB[0] = delta
        dJdW[0] = X @ delta.T
        return (dJdB, dJdW)
    
def calc_loss(X_batch: np.ndarray, y_batch: np.ndarray, weights, biases, activation: np.ufunc) -> float:
    loss = 0.0
    for X, y in zip(X_batch, y_batch):
        curr = X
        for W, b in zip(weights, biases):
            z = W.T @ curr + b
            curr = activation(z)
        loss += np.sum((curr - y) ** 2)
    return loss