import numpy as np


class Error:
    
    def fun(self, yTrue : np.ndarray, yPred : np.ndarray) -> np.ndarray:
        pass
    def fun_prime(self, yTrue : np.ndarray, yPred : np.ndarray) -> np.ndarray:
        pass