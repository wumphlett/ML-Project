import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Sequence
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import sparse
import warnings

# I hate overflow errors :)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in exp')

def sigmoid(X: np.ndarray) -> np.ndarray:
    '''Sigmoid function'''
    return 1.0 / (1.0 + np.exp(-X))
def dSigmoid(X: np.ndarray) -> np.ndarray:
    '''Derivative of sigmoid function'''
    a = 1.0 / (1.0 + np.exp(-X))
    return a*(1-a)

def tanh(X: np.ndarray) -> np.ndarray:
    '''Tanh function'''
    return np.tanh(X)
def dTanh(X: np.ndarray) -> np.ndarray:
    '''Derivative of tanh function'''
    return 1.0 - np.tanh(X)**2

def relu(X: np.ndarray) -> np.ndarray:
    '''Rectified linear unit function'''
    return np.maximum(0, X)
def dRelu(X: np.ndarray) -> np.ndarray:
    '''Derivative of rectified linear unit function'''
    return np.where(X > 0, 1, 0)

def softmax(X: np.ndarray) -> np.ndarray:
    tmp = X - X.max(axis=1)[:, np.newaxis]
    return np.exp(tmp) / np.exp(tmp).sum(axis=1)[:, np.newaxis]

def squared_loss(y_true: np.ndarray, y_pred: np.ndarray):
    return ((y_true - y_pred) ** 2).mean() / 2

def L1_reg_loss(weights: list[np.ndarray]) -> float:
    c = 0
    for w in weights:
        c += np.sum(np.abs(w))
    return c

def L1_reg_grad(weights: list[np.ndarray]) -> list[np.ndarray]:
    grad = []
    for w in weights:
        grad.append(np.where(w > 0, 1, -1))
    return grad

def L2_reg_loss(weights: list[np.ndarray]) -> float:
    c = 0
    for w in weights:
        w = w.ravel()
        c += np.dot(w, w)
    return c

def L2_reg_grad(weights: list[np.ndarray]) -> list[np.ndarray]:
    return [2*w for w in weights ]

class MultiLayerPerceptron:
    ''' 
    An implementation of a multi-layer perceptron with backpropagation 

    Parameters
    -----------
    epochs: int
        Number of epochs to train the model
    lr: float
        Learning rate
    hidden_layers: Sequence[int]
        A sequence of integers representing the number of neurons in each hidden layer
    regularization: str, default=None
        The regularization function to use. Can be either None, "l1" or "l2"
    reg_const: float, default=0.0
        The regularization constant
    activation: str, default="sigmoid"
        The activation function to use. Can be either "sigmoid", "tanh" or "relu"
    '''

    def __init__(
        self,
        epochs: int,
        lr: float,
        hidden_layers: Sequence[int],
        regularization: str = None,
        reg_const: float = 0.0,
        activation: str = "sigmoid",
    ):
        self.num_epochs = epochs
        self.lr = lr
        self.regularization = regularization
        self.reg_const = reg_const
        self.hidden_layers = hidden_layers
        self.output_layer = None
        self.input_layer = None
        self._num_layers = len(self.hidden_layers) + 2
        self._yencoder = LabelBinarizer()

        self._output_activation = None

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
        
        match regularization:
            case None:
                pass
            case "l1" | "L1":
                self._loss_reg = L1_reg_loss
                self._grad_reg = L1_reg_grad
            case "l2" | "L2":
                self._loss_reg = L2_reg_loss
                self._grad_reg = L2_reg_grad
            case _:
                raise ValueError("Invalid regularization function")

        self._loss_function = squared_loss
        self._biases = None
        self._weights = None 
        # self._biases = [np.zeros((y, 1)) for y in self._structure[1:]]
        # self._weights = [np.zeros((x,y)) for x, y in zip(self._structure[:-1], self._structure[1:])]
        self.train_loss_curve = []
        self.val_loss_curve = []
    
    def epochs(self):
        for i in range(self.num_epochs):
            yield i, self.lr
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray = None, y_val:np.ndarray = None, batch_size: int = 1, continue_fit = False) -> None:
        ''' 
        Fits the model to the given data

        Parameters
        -----------
        X: np.ndarray
            Input data of shape (n_examples, n_features)
        y: np.ndarray
            Output data of shape (n_examples, )
        X_val: np.ndarray, optional
            Validation input data of shape (n_examples, n_features)
        y_val: np.ndarray, optional
            Validation output data of shape (n_examples, )
        batch_size: int, default=1
            Size of the batch to be used for training
        continue_fit: bool, default=False
            If True, the model will continue training from the last epoch
        '''
        if len(X.shape) != 2:
            raise ValueError("Invalid shape for X")
        if len(y.shape) != 1:
            raise ValueError("Invalid shape for y")
        n_examples, n_features = X.shape

        use_val = False
        if X_val is not None and y_val is not None:
            use_val = True
            if len(X_val.shape) != 2:
                raise ValueError("Invalid shape for X_val")
            if len(y_val.shape) != 1:
                raise ValueError("Invalid shape for y_val")
        y_combined = np.concatenate((y, y_val)) if use_val else y
        y_combined = self._format_labels(y_combined)
        y, y_val = np.split(y_combined, [n_examples]) if use_val else (y_combined, None)
        
        if not continue_fit:
            self.input_layer = n_features
            self.output_layer = y.shape[1]
            self._structure = (self.input_layer, *self.hidden_layers, self.output_layer)

            self._biases = [np.random.randn(y, 1) for y in self._structure[1:]]
            self._weights = [np.random.randn(x,y) for x, y in zip(self._structure[:-1], self._structure[1:])]

        for epoch_num, lr in tqdm(self.epochs(), total=self.num_epochs):
            train_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                dJdB, dJdW, c_loss = self._backprop(X_batch, y_batch)
                train_loss += c_loss

                self._biases = [b - lr * db for b, db in zip(self._biases, dJdB)]
                self._weights = [w - lr * dw for w, dw in zip(self._weights, dJdW)]
            num_batches = X.shape[0] // batch_size
            self.train_loss_curve.append(train_loss / num_batches)
            if use_val:
                val_loss = self._calc_loss(y_val, self._fast_forward_pass(X_val))
                self.val_loss_curve.append(val_loss)


    def predict(self, X:np.ndarray) -> np.ndarray:
        '''
        Predicts the output for the given input
        X: Input data of shape (n_examples, n_features)
        returns: Output data of shape (n_examples, )
        '''
        curr = self._fast_forward_pass(X)
        if self.output_layer == 1:
            curr = curr.ravel()
        return self._yencoder.inverse_transform(curr)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Returns the accuracy of the model on the given data
        X: Input data of shape (n_examples, n_features)
        y: Output data of shape (n_examples, )
        '''
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def _format_labels(self, y: np.ndarray) -> np.ndarray:
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif len(y.shape) == 2 and y.shape[1] > 1 or len(y.shape) > 2:
            raise ValueError("Invalid shape for y")

        self._yencoder.fit(y)
        if len(self._yencoder.classes_) == 2:
            self._output_activation = sigmoid
        else:
            self._output_activation = softmax
        return self._yencoder.transform(y)

    def _backprop(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # initialize empty lists to store the gradient of the biases and weights
        dBias = [np.zeros(b.shape) for b in self._biases]
        dWeights = [np.zeros(w.shape) for w in self._weights]
        n_samples = X.shape[0]

        # do a forward pass to get the activations and z values
        layer_raw = [] # stores the weighted sum of inputs for each layer
        layer_activations = [] # stores the output of each layer
        a = X # input layer
        for i, (b, W) in enumerate(zip(self._biases, self._weights)):
            # compute the weighted sum of inputs for this layer
            z = a @ W + b.T

            # apply the activation function to the output of this layer
            if i < self._num_layers - 2:
                a = self.activation(z)
            else:
                a = self._output_activation(z)
            # store the raw output and the activated output of this layer
            layer_raw.append(z)
            layer_activations.append(a)
        
        # calculate the loss
        loss = self._calc_loss(y, a)

        # index of the last hidden layer
        last_hidden = self._num_layers - 2

        # compute the error at the last hidden layer
        delta = (layer_activations[last_hidden] - y)

        # compute the gradient of the biases and weights for the last hidden layer
        dBias[last_hidden] = np.mean(delta, axis=0)
        dWeights[last_hidden] = layer_activations[last_hidden-1].T @ delta

        # for all hidden layers except the first and last,
        # compute the gradient of the biases and weights
        for L in range(last_hidden-1, 0, -1):
            # compute the error at this layer
            delta = (delta @ self._weights[L+1].T) * self.dActivation(layer_raw[L])

            # compute the gradient of the biases and weights for this layer
            dBias[L] = np.mean(delta, axis=0)
            dWeights[L] = layer_activations[L-1].T @ delta

        # for the input layer, compute the gradient of the biases and weights
        # using the inputs, rather than the previous layer
        delta = (delta @ self._weights[1].T) * self.dActivation(layer_raw[0])
        dBias[0] = np.mean(delta, axis=0)
        dWeights[0] = X.T @ delta

        # add a second dimension to the bias gradient to make it compatible with the bias shape
        dBias = [db[:,np.newaxis] for db in dBias]
        # divide by number of samples to get the average gradient
        dWeights = [dw/n_samples for dw in dWeights]

        # apply regularization if enabled
        if self.regularization:
            dWeights = [dw + self.reg_const * r for dw, r in zip(dWeights, self._grad_reg(self._weights))]
        return (dBias, dWeights, loss)
    
    def _calc_loss(self, y_pred: np.ndarray, y_batch: np.ndarray) -> float:
        loss = self._loss_function(y_pred, y_batch)
        if self.regularization:
            loss += (0.5 * self.reg_const) * self._loss_reg(self._weights) / y_pred.shape[0]
        return loss

    def _fast_forward_pass(self, X: np.ndarray) -> np.ndarray:
        curr = X
        for i in range(self._num_layers - 1):
            curr = curr @ self._weights[i]
            curr += self._biases[i].T
            if i < self._num_layers - 2:
                curr = self.activation(curr)
            else:
                curr = self._output_activation(curr)
        return curr
    
    def plot_loss(self) -> None:
        plt.plot(self.train_loss_curve, label='Training loss')
        if self.val_loss_curve and len(self.val_loss_curve) == len(self.train_loss_curve):
            plt.plot(self.val_loss_curve, label='Validation loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.show()