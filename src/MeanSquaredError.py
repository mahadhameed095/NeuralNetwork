from Error import Error
import numpy as np
class MeanSquaredError(Error):
    
    def fun(self, yTrue: np.array, yPred: np.array) -> np.array:
        return np.mean(np.power(yTrue - yPred, 2), 0)
        
    def fun_prime(self, yTrue: np.array, yPred: np.array) -> np.array:
        return 2 * (yPred - yTrue) / yPred.shape[1] 

    