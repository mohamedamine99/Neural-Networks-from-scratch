# TODO: add type annotations and return types
# TODO: check numerical values are correct


import numpy as np

def mse(y_true, y_pred):
    diff = y_pred - y_true
    squared_diff = np.square(diff)
    return np.mean(squared_diff)

def mse_backward(y_true, y_pred):
    diff = y_pred - y_true
    return 2 * diff / np.size(y_true)

