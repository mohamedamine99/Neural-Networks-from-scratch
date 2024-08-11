import numpy as np

from layers import BaseLayer


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


