import numpy as np
from layers import BaseLayer
from losses_functional import  mse, mse_backward

class BaseLoss(BaseLayer):

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class MSE(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return mse(y_true, y_pred)


    def backward(self, y_true, y_pred):
        return mse_backward(y_true, y_pred)


