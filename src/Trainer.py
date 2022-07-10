import numpy as np
from Error import Error
from Network import Network
import Utils
class Trainer():

    def __init__(self, netLayers : Network) -> None:
        self._netLayers = netLayers._layers

    def forward(self, input) -> np.ndarray:
        for layer in self._netLayers:
            input = layer.forward(input)
        return input

    def backward(self, grad) -> np.ndarray:
        for layer in reversed(self._netLayers):
            grad = layer.backward(grad)
        return grad


    def train(self, epochs : int, batch_size : int, train_x : np.ndarray, train_y : np.ndarray,  cost : Error, calcAccuracy) -> None:
        assert train_x.shape[1] == train_y.shape[1], "The number of samples in x and y are different."
        for i in range(epochs):
            for (batchX, batchY) in zip(np.split(train_x, np.arange(batch_size, train_x.shape[1], batch_size), 1), np.split(train_y, np.arange(batch_size, train_y.shape[1], batch_size), 1)):
                forward = self.forward(batchX)
                grad = cost.fun_prime(batchY, forward)
                backward = self.backward(grad)
            predictions = self.forward(train_x)
            # error = np.mean(cost.fun(train_y, predictions), 0)
            accuracy = calcAccuracy(predictions, train_y)
            Utils.printProgressBar(i, epochs, suffix = (str(accuracy)+"%"))
        
