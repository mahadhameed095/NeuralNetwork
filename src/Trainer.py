import numpy as np
from Error import Error
from Network import Network
import Utils
class Trainer():

    def __init__(self, netLayers : Network) -> None:
        self._netLayers = netLayers._layers

    def forward(self, input) -> np.array:
        for layer in self._netLayers:
            input = layer.forward(input)
        return input

    def backward(self, grad) -> np.array:
        for layer in reversed(self._netLayers):
            grad = layer.backward(grad)
        return grad


    def train(self, epochs : int, batch_size : int, train_x : np.array, train_y : np.array,  cost : Error) -> None:
        for i in range(epochs):
            for (batchX, batchY) in zip(np.split(train_x, np.arange(batch_size, train_x.shape[1], batch_size), 1), np.split(train_y, np.arange(batch_size, train_y.shape[1], batch_size), 1)):
                forward = self.forward(train_x)
                grad = cost.fun_prime(train_y, forward)

            
                backward = self.backward(grad)
            error = np.mean(cost.fun(train_y, forward), 0)
            Utils.printProgressBar(i, epochs, suffix=str(error))
        
