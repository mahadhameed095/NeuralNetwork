from Error import Error
import numpy as np

class BinaryCrossEntropy(Error):
    
    def fun(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return -1 * np.mean((yTrue * np.log(yPred)) + ((1 - yTrue) * np.log(1 - yPred)) , 0)

    def fun_prime(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return (-1 / yTrue.shape[1]) * ((yTrue / yPred) - ((1 - yTrue) / (1 - yPred)))
