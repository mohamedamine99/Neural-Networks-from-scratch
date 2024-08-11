# TODO: add type annotations and return types
# TODO: check numerical values are correct


import numpy as np

def mse(y_true, y_pred):
    diff = y_pred - y_true
    squared_diff = np.square(diff)
    return np.mean(squared_diff)

def mse_backward(y_true, y_pred):
    # print('---- MSE backward ----')
    # print(f'{y_true.shape=}')
    # print(f'{y_pred.shape=}')

    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    # Clip y_pred to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    # Clip y_pred to avoid division by zero
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
