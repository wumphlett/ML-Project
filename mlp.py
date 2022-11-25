import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Sequence
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse

def sigmoid(X: np.ndarray):
    return 1.0 / (1.0 + np.exp(-X))
def dSigmoid(X: np.ndarray):
    a = 1.0 / (1.0 + np.exp(-X))
    return a*(1-a)

def tanh(X: np.ndarray):
    return np.tanh(X)
def dTanh(X: np.ndarray):
    return 1.0 - np.tanh(X)**2

def relu(X: np.ndarray):
    return np.maximum(0, X)
def dRelu(X: np.ndarray):
    return np.where(X > 0, 1, 0)

def squared_loss(y_true: np.ndarray, y_pred: np.ndarray):
    return 0.5 * np.sum((y_true - y_pred)**2)

class MultiLayerPerceptron:
    def __init__(
        self,
        epochs: int,
        lr: float,
        hidden_layers: Sequence[int],
        reg_const: float = 0.0,
        activation: str = "sigmoid",
    ):
        self.num_epochs = epochs
        self.lr = lr
        self.reg_const = reg_const
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = hidden_layers
        # self._structure = (input_layer, *hidden_layers, self.output_layer)
        self._num_layers = len(self.hidden_layers) + 2
        self._yencoder = LabelBinarizer()

        match activation:
            case "sigmoid" | "logistic":
                self.activation = sigmoid
                self.dActivation = dSigmoid
            case "tanh":
                self.activation = tanh
                self.dActivation = dTanh
            case "relu":
                self.activation = relu
                self.dActivation = dRelu
            case _:
                raise ValueError("Invalid activation function")

        self._loss_function = squared_loss

        self._biases = None
        self._weights = None 
        # self._biases = [np.zeros((y, 1)) for y in self._structure[1:]]
        # self._weights = [np.zeros((x,y)) for x, y in zip(self._structure[:-1], self._structure[1:])]
        self.loss_curve = []
        self.fitted = False
    
    def epochs(self):
        for i in range(self.num_epochs):
            yield i, self.lr
    
    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1) -> None:
        ''' Fits the model to the given data
        X: Input data of shape (n_examples, n_features)
        y: Output data of shape (n_examples, )
        '''

        if len(X.shape) != 2:
            raise ValueError("Invalid shape for X")
        n_examples, n_features = X.shape
        self.input_layer = n_features

        y = self._format_labels(y)
        self.output_layer = y.shape[1]
        
        self._structure = (self.input_layer, *self.hidden_layers, self.output_layer)

        self._biases = [np.random.randn(y, 1) for y in self._structure[1:]]
        self._weights = [np.random.randn(x,y) for x, y in zip(self._structure[:-1], self._structure[1:])]


        for epoch_num, lr in tqdm(self.epochs(), total=self.num_epochs):
            loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                dJdB, dJdW, c_loss= self._backprop(X_batch, y_batch)
                loss += c_loss
                self._biases = [b - lr * db for b, db in zip(self._biases, dJdB)]
                self._weights = [w - lr * dw for w, dw in zip(self._weights, dJdW)]
            self.loss_curve.append(loss)

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''
        Predicts the output for the given input
        X: Input data of shape (n_examples, n_features)
        returns: Output data of shape (n_examples, )
        '''
        curr = X
        for i in range(self._num_layers - 1):
            curr = curr @ self._weights[i]
            # curr = self._safe_sparse_dot(curr,self._weights[i])
            curr += self._biases[i].T
            curr = self.activation(curr)
        if self.output_layer == 1:
            curr = np.where(curr > 0.5, 1, 0).ravel()
        else:
            curr = np.argmax(self.activation(curr), axis=1).ravel()
        return curr
    
    def _format_labels(self, y: np.ndarray) -> np.ndarray:
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif len(y.shape) == 2 and y.shape[1] > 1 or len(y.shape) > 2:
            raise ValueError("Invalid shape for y")

        if self.output_layer == 1:
            return y[:, np.newaxis]
        else:
            return self._yencoder.fit_transform(y)

    def _backprop(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        dBias = [np.zeros(b.shape) for b in self._biases]
        dWeights = [np.zeros(w.shape) for w in self._weights]
        n_samples = X.shape[0]

        # do forward pass, store all activations and net values
        layer_raw = []
        layer_activations = []
        a = X
        for b, W in zip(self._biases, self._weights):
            z = a @ W + b.T
            # z = self._safe_sparse_dot(a,W) + b.T
            a = self.activation(z)
            layer_raw.append(z)
            layer_activations.append(a)
        
        loss = self._calc_loss(y, a)

        # For last layer, compare to y
        last_hidden = self._num_layers - 2

        delta = (layer_activations[last_hidden] - y) * self.dActivation(layer_raw[last_hidden])
        dBias[last_hidden] = np.mean(delta, axis=0)
        dWeights[last_hidden] = layer_activations[last_hidden-1].T @ delta
        # dWeights[last_hidden] = self._safe_sparse_dot(layer_activations[last_hidden-1].T, delta)
        dWeights[last_hidden] /= n_samples

        # For all hidden layers, compare to layer after it
        for L in range(last_hidden-1, 0, -1):
            delta = (delta @ self._weights[L+1].T) * self.dActivation(layer_raw[L])
            # delta = self._safe_sparse_dot(delta, self._weights[L+1].T) * self.dActivation(layer_raw[L])
            dBias[L] = np.mean(delta, axis=0)
            dWeights[L] = layer_activations[L-1].T @ delta

        # For input layer, update W according to input, not previous layer
        delta = (delta @ self._weights[1].T) * self.dActivation(layer_raw[0])
        # delta = self._safe_sparse_dot(delta, self._weights[1].T) * self.dActivation(layer_raw[0])
        dBias[0] = np.mean(delta, axis=0)
        dWeights[0] = X.T @ delta
        # add a second dimension to the bias gradient to make it compatible with the bias shape
        dBias = [db[:,np.newaxis] for db in dBias]
        return (dBias, dWeights, loss)
    
    def _calc_loss(self, y_pred: np.ndarray, y_batch: np.ndarray, reg:str=None) -> float:
        loss = self._loss_function(y_pred, y_batch)
        if reg:
            match reg:
                case "l1":
                    c = 0
                    for w in self._weights:
                        c += np.sum(np.abs(w))
                    loss += self.reg_const * c
                case "l2":
                    c = 0
                    for w in self._weights:
                        c += np.sum(w**2)
                    loss += self.reg_const * c
                case _:
                    raise ValueError("Invalid regularization method")
        return loss

    def _safe_sparse_dot(self, a, b, dense_output=False):
        """Dot product that handle the sparse matrix case correctly
        Lovingly copied from sklearn.utils.extmath.safe_sparse_dot
        """
        if a.ndim > 2 or b.ndim > 2:
            if sparse.issparse(a):
                # sparse is always 2D. Implies b is 3D+
                # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
                b_ = np.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            elif sparse.issparse(b):
                # sparse is always 2D. Implies a is 3D+
                # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            else:
                ret = np.dot(a, b)
        else:
            ret = a @ b
        
        if (
            sparse.issparse(a)
            and sparse.issparse(b)
            and dense_output
            and hasattr(ret, "toarray")
        ):
            return ret.toarray()
        return ret
    
    def plot_loss(self) -> None:
        plt.plot(self.loss_curve)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.show()

if __name__ == "__main__":
    from TwoDimProblem import TwoDimProblem
    p = TwoDimProblem(value_range=5)
    X, y = p.createData(soln_rank=2, noise_frac=0.0, samples=10)
    y[:2] = 2 
    
    mlp = MultiLayerPerceptron(
        epochs=1, 
        lr=0.1,
        activation='sigmoid',
        # input_layer=2, 
        hidden_layers=[3,2],
        # output_layer=2
        )
        
    mlp.fit(X,y, batch_size=5)
    # p.plotPred(pred, show_correct=True)
    # mlp.plot_loss()