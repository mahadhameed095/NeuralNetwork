from Layer import Layer
import numpy as np
class Sigmoid(Layer):
    
    def __init__(self) -> None:
        self._trainable = False
        self.input = None

    def forward(self, input: np.array) -> np.array:
        self.input = np.clip(1/(1+np.exp(-input)), 0.0001, 0.9999)
        return self.input

    def backward(self, outputGradient: np.array) -> np.array:
        return outputGradient * (self.input * (1 - self.input))