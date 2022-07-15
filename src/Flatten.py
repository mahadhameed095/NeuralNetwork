import numpy as np
from Layer import Layer

"""
    This Layer is meant to transform a 3 dimensional array, (K, M, N) to a 2 dimensional array of (M*N, K), and back as well.
    It is required between a convolutional unit and a fully connected layer / dense layer.
"""
class Flatten(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._input_shape = None
    
    def _initLayer(self, argsDict: dict) -> dict:
        assert "input_shape" in argsDict, "Input Shape not specified"
        _, _, width, height = argsDict["input_shape"]
        argsDict["input_shape"] = width * height
        return argsDict

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input_shape = input.shape
        return input.reshape(input.shape[0], input.shape[2] * input.shape[3]).T

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        return outputGradient.T.reshape(self._input_shape[0], 1, self._input_shape[2], self._input_shape[3])
