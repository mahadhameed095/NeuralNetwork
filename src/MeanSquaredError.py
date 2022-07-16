from Error import Error
import numpy as np
class MeanSquaredError(Error):
    
    def fun(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return np.mean(np.power(yTrue - yPred, 2), 0)
        
    def fun_prime(self, yTrue: np.ndarray, yPred: np.ndarray) -> np.ndarray:
        return 2 * (yPred - yTrue) / yPred.shape[0] 

    