import numpy as np
from layers import BaseLayer


class BaseLoss(BaseLayer):

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class RMSE(BaseLoss):
    pass




