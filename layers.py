# TODO: add attributes initialisation to avoid errors


import numpy
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
        self.W = np.random.uniform(size=(out_features, in_features)) * 0.01
        self.add_bias = add_bias
        if self.add_bias:
            self.B = np.random.uniform(size=(out_features, 1)) * 0.01


    def forward(self, input:numpy.ndarray):
        self.input = input
        self.output = np.matmul(self.W, input) + self.B
        return np.copy(self.output)

    def backward(self, output_grad: numpy.ndarray):
        self.input_grad = np.matmul(self.W.T, output_grad)
        self.w_grad = np.matmul(output_grad, self.input)
        if self.add_bias:
            self.b_grad = output_grad.copy()
        return np.copy(self.input_grad)


    def update_params(self, learning_rate:float = 0.1):
        self.W -= learning_rate * self.w_grad
        if self.add_bias:
            self.B -= learning_rate * self.b_grad

