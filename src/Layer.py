import numpy as np

class Layer:
    def __init__(self) -> None:
        self._trainable = False
        pass
    
    def forward(self, input : np.array) -> np.array:
        pass
    
    def backward(self, outputGradient: np.array) -> np.array:
        pass
