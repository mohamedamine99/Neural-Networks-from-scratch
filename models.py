from typing import List
import numpy as np


class BaseModel:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def set_train_mode(self):
        pass

    def set_eval_mode(self):
        pass




class Sequential(BaseModel):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


    def forward(self, input:np.ndarray):
        for layer in self.layers:
            input = layer.forward(input)

    def backward(self, output_grad:np.ndarray):
        for layer in self.layers.reverse():
            output_grad = layer.forward(output_grad)
