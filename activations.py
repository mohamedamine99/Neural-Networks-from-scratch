
# TODO: add attributes initialisation to avoid errors

import numpy as np

from activations_functional import (relu, relu_backward,
                                    tanh, tanh_backward,
                                    sigmoid, sigmoid_backward,

                                    )

from layers import BaseLayer

class Relu(BaseLayer):

    def forward(self, X:np.ndarray) -> np.ndarray:
        self.Y = relu(X)
        return self.Y

    def backward(self, output_grad:np.ndarray) -> np.ndarray:
        self.input_grad = relu_backward(output_grad)
        return self.input_grad


class Tanh(BaseLayer):

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Y = tanh(X)
        return self.Y

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.input_grad = tanh_backward(output_grad)
        return self.input_grad


class Sigmoid(BaseLayer):

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Y = sigmoid(X)
        return self.Y

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.input_grad = sigmoid_backward(output_grad)
        return self.input_grad




