import numpy as np


class Error:
    
    def fun(self, yTrue : np.array, yPred : np.array) -> np.array:
        pass
    def fun_prime(self, yTrue : np.array, yPred : np.array) -> np.array:
        pass