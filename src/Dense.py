import numpy as np
from Layer import Layer

class Dense(Layer):
    
    def __init__(self, inputLayerSize, outputLayerSize, learning_rate) -> None:
        super().__init__()
        self._alpha = learning_rate
        self._input = None
        self._weights = np.random.randn(outputLayerSize, inputLayerSize)
        self._bias = np.random.randn(outputLayerSize, 1)
        self._trainable = True

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input = input
        return (self._weights @ input) + self._bias
    
    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        weightGradient = outputGradient @ self._input.T
        inputGradient = self._weights.T @ outputGradient
        self._weights = self._weights - self._alpha * weightGradient
        self._bias = self._bias - self._alpha * np.sum(outputGradient, axis = 1, keepdims=True)
        return inputGradient

