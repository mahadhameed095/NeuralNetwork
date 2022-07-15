import numpy as np
from Layer import Layer
class Softmax(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._input = np.exp(input - np.max(input, 0))
        self._input = np.clip((self._input / np.sum(self._input, 0)), 0.0001, 0.9999)
        return self._input
    
    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        grads = np.empty(self._input.shape, dtype=np.float32)
        for i in range(grads.shape[1]):
            grads[:, [i]] = (np.diag(self._input[:, i]) - (self._input[:, [i]] @ self._input[:, [i]].T)) @ outputGradient[:, [i]]
        return grads