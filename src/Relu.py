import numpy as np
from Layer import Layer


class Relu(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input = np.max(input, 0)
        return self._input

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        return np.multiply(outputGradient, np.heaviside(self._input, 0.5))
