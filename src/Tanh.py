import numpy as np
from Layer import Layer

class Tanh(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input = np.tanh(input)
        return self._input
    
    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        return np.multiply(outputGradient, 1 - np.power(self._input, 2))