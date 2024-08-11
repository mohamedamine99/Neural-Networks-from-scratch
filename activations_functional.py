# TODO : add softmax activation function


import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1., 0.)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.pow(tanh(x), 2)


def softmax(x):
    # the max term is computer for added numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_prime(x):
    # s is the output of the softmax function
    s = softmax(x)
    # Compute the Jacobian matrix
    jacobian_matrix = np.diag(s) - np.outer(s, s)
    return jacobian_matrix