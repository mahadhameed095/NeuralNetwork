import numpy as np
from Layer import Layer

class Dense(Layer):
    
    def __init__(self, num_neurons : int, learning_rate : float = None) -> None:
        super().__init__()
        self._alpha = learning_rate
        self._input = None
        self._trainable = True
        self._num_neurons = num_neurons
        self._weights = None
        self._bias = None
        
    def _initLayer(self, argsDict: dict) -> dict:
        inputLayerSize = argsDict["input_shape"]
        self._weights = np.random.randn(self._num_neurons, inputLayerSize)
        self._bias = np.random.randn(self._num_neurons, 1)
        if self._alpha == None:
            assert "learning_rate" in argsDict, "Learning Rate not specified. Either specify in the layer constructor, or pass as argument in dictionary"
            self._alpha = argsDict["learning_rate"]
        argsDict["input_shape"] = self._num_neurons
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

