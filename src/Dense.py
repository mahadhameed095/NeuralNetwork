from operator import le
import numpy as np
from Layer import Layer

class Dense(Layer):
    
    def __init__(self) -> None:
        super().__init__()
        self._alpha = None
        self._input = None
        self._trainable = True
        self._weights = None
        self._bias = None
        
    def _initLayer(self, num_neurons : int, argsDict: dict, learning_rate : float = None) -> dict:
        inputLayerSize = argsDict["input_shape"]
        argsDict["input_shape"] = num_neurons
        self._weights = np.random.randn(num_neurons, inputLayerSize)
        self._bias = np.random.randn(num_neurons, 1)
        if learning_rate == None:
            assert argsDict["learning_rate"] is not None, "Learning Rate not specified. Either specify in the layer constructor, or pass as argument in the network constructor"
            self._alpha = argsDict["learning_rate"]
        else:
            self._alpha = learning_rate
        return argsDict
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input = input
        return (self._weights @ input) + self._bias
    
    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        weightGradient = outputGradient @ self._input.T
        inputGradient = self._weights.T @ outputGradient
        self._weights = self._weights - self._alpha * weightGradient
        self._bias = self._bias - self._alpha * np.sum(outputGradient, axis = 1, keepdims=True)
        return inputGradient

