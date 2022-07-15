import numpy as np
from Layer import Layer


class Relu(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        # self._input = ((input > 0) * input) + ((input <= 0) * 0.5 * input)
        self._input = input * (input > 0)
        return self._input

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        # return outputGradient * (self._input > 0) + ((self._input <= 0) * 0.5)
        return outputGradient * (1. * (self._input > 0))
