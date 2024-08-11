import numpy as np

from layers import BaseLayer
from activations_functional import (relu, relu_prime,
                                    tanh, tanh_prime,
                                    sigmoid, sigmoid_prime,
                                    softmax, softmax_prime
                                    )


class BaseActivation(BaseLayer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input:np.ndarray):
        self.input = np.copy(input)
        self.output = self.activation(self.input)
        return np.copy(self.output)



    def backward(self, output_grad):
        self.output_grad = np.copy(output_grad)
        self.input_grad = np.multiply(output_grad, self.activation_prime(self.input))
        return np.copy(self.input_grad)


class ReLU(BaseActivation):
    def __init__(self):
        super().__init__(relu, relu_prime)


class Tanh(BaseActivation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


class Sigmoid(BaseActivation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class Softmax(BaseActivation):
    def __init__(self):
        super().__init__(softmax, softmax_prime)


