import numpy as np
from Layer import Layer

"""
    This Layer is meant to transform a 4 dimensional array, (B, C, M, N) to a 2 dimensional array of (C*M*N, B), and back as well.
    It is required between a convolutional unit and a fully connected layer / dense layer.
"""
class Flatten(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._input_shape = None
    
    def _initLayer(self, argsDict: dict) -> dict:
        _, _, size = argsDict["input_shape"]
        argsDict["input_shape"] = size
        return argsDict

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input_shape = input.shape
        return input.reshape(input.shape[0], input.shape[1] * input.shape[2])

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        return outputGradient.reshape(*self._input_shape)
