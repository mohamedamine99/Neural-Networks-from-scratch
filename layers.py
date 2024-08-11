# TODO: add attributes initialisation to avoid errors
# TODO: complete type annotations and return types
# TODO: Add documentations

from typing import Tuple
import numpy as np

# print(np.__version__)

class BaseLayer:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass



class Linear(BaseLayer):
    def __init__(self, in_features:int, out_features:int, add_bias:bool=True):
        self.in_features = in_features
        self.out_features = out_features
        self.add_bias = add_bias
        self.W = np.random.uniform(size=(out_features, in_features)) * 0.01
        self.B = np.random.uniform(size=(out_features, 1)) * 0.01 if add_bias else None
        self.input = None
        self.output = None
        self.input_grad = None
        self.w_grad = None
        self.b_grad = None

    def forward(self, input:np.ndarray):
        # print('----- FORWARD -----')
        self.input = np.copy(input)
        # print(f'{self.input.shape=}')
        # print(f'{self.W.shape=}')
        self.output = np.dot(self.W, self.input) + self.B
        # print(f'{self.output.shape=}')
        # print('---------------------')
        return np.copy(self.output)

    def backward(self, output_grad: np.ndarray, learning_rate):
        # print('------ BACKWARD ------')
        # print(f'{output_grad.shape=}')
        # print(f'{self.W.T.shape=}')
        # print(f'{self.input.T.shape=}')
        self.input_grad = np.dot(self.W.T, output_grad)
        self.w_grad = np.dot(output_grad, self.input.T)
        if self.add_bias:
            self.b_grad = output_grad.copy()

        # print(f'{self.w_grad.shape=}')
        # print(f'{self.b_grad.shape=}')
        # print('-----------------------')

        self.update_params(learning_rate)
        return np.copy(self.input_grad)


    def update_params(self, learning_rate:float = 0.1):
        self.learning_rate = learning_rate
        self.W -= learning_rate * self.w_grad
        if self.add_bias:
            self.B -= learning_rate * self.b_grad



class Reshape(BaseLayer):
    def __init__(self, shape:Tuple[int, ...]):
        super().__init__()
        self.output_shape = shape

    def forward(self, input:np.ndarray):
        self.input = np.copy(input)
        self.input_shape = self.input.shape

        return np.reshape(input, newshape=self.output_shape)


    def backward(self, output: np.ndarray):
        return np.reshape(output, newshape=self.input_shape)
