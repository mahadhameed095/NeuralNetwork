import numpy as np

class Layer:
    def __init__(self) -> None:
        self._trainable = False
        pass
    
    def forward(self, input : np.ndarray) -> np.ndarray:
        pass
    
    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        pass
