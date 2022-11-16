import numpy as np
from numba import njit, jit, float64
from numba.typed import List

@njit()
def backprop_compiled( 
        X: np.ndarray, 
        y: np.ndarray,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        activation: np.ufunc,
        dActivation: np.ufunc,
        structure: tuple) -> tuple[list[np.ndarray], list[np.ndarray]]:

        # print(f"X: {X.shape}")
        # print(f"y: {y.shape}")
        # print(f"weights: {[w.shape for w in weights]}")
        # print(f"biases: {[b.shape for b in biases]}")
        

        dJdB = [np.zeros(b.shape) for b in biases]
        dJdW = [np.zeros(w.shape) for w in weights]
        
        layer_raw = [np.zeros(b.shape) for b in biases]
        layer_activations = [np.zeros(b.shape) for b in biases]
        a = X
        for i, (b, W) in enumerate(zip(biases, weights)):
            z = W.T @ a + b 
            a = activation(z)
            # print(f"z{i}: {z.shape}")
            # print(f"a{i}: {a.shape}")
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

@jit(nopython=True)
def batch_backprop_compiled(X_batch: np.ndarray, 
    y_batch: np.ndarray, 
    weights: list[np.ndarray], 
    biases: list[np.ndarray], 
    activation: np.ufunc, 
    dActivation: np.ufunc, 
    structure: tuple,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:

    dJdB = [np.zeros(b.shape) for b in biases]
    dJdW = [np.zeros(w.shape) for w in weights]
    batch_size = len(X_batch)
    for X, y in zip(X_batch, y_batch):
        dB, dW = backprop_compiled(X, y, weights, biases, activation, dActivation, structure)
        dJdB = [db + ddb for db, ddb in zip(dJdB, dB)]
        dJdW = [dw + ddw for dw, ddw in zip(dJdW, dW)]
    dJdB = [db / batch_size for db in dJdB]
    dJdW = [dw / batch_size for dw in dJdW]
    return (dJdB, dJdW)

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
    batch_size = len(X_batch)
    for X, y in zip(X_batch, y_batch):

        X = X[:, np.newaxis]
        y = y[:, np.newaxis]

        weights = List(weights)
        biases = List(biases)

        dB, dW = backprop_compiled(X, y, weights, biases, activation, dActivation, structure)


        dJdB = [db + ddb for db, ddb in zip(dJdB, dB)]
        dJdW = [dw + ddw for dw, ddw in zip(dJdW, dW)]
    dJdB = [db / batch_size for db in dJdB]
    dJdW = [dw / batch_size for dw in dJdW]
    return (dJdB, dJdW)
