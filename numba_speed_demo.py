from numba import njit, guvectorize, float64
import numpy as np
from timeit import timeit

def normal_numpy_sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

@guvectorize([(float64[:], float64[:])], "(n)->(n)")
def numba_numpy_sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

x = np.random.rand(100_000)

print(timeit(lambda: normal_numpy_sigmoid(x), number=100))

numba_numpy_sigmoid(x)

print(timeit(lambda: numba_numpy_sigmoid(x), number=100))