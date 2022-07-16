from inspect import Parameter
import numpy as np
from Error import Error
from Network import Network
from math import floor
import Utils
class Trainer():

    def __init__(self, netLayers : Network) -> None:
        self._netLayers = netLayers
        

    def _forward(self, input : np.ndarray) -> np.ndarray:
        # print(input.shape)
        for layer in self._netLayers._layers:
            input = layer.forward(input)
            # print(str(type(layer))+"->", input.shape)
        return input

    def _backward(self, grad : np.ndarray) -> np.ndarray:
        for layer in reversed(self._netLayers._layers):
            grad = layer.backward(grad)
        return grad

    def train(self, epochs : int, batch_size : int, train_x : np.ndarray, train_y : np.ndarray,  cost : Error, calcAccuracy) -> None:
        assert train_x.shape[0] == train_y.shape[0], "The number of samples in x and y are different."
        steps = floor(train_x.shape[0] / batch_size)
        for i in range(epochs):
            # for (batchX, batchY) in zip(np.split(train_x, np.arange(batch_size, train_x.shape[1], batch_size), 0), np.split(train_y, np.arange(batch_size, train_y.shape[1], batch_size), 0)):
            pivot = 0
            for j in range(steps):
                forward = self._forward(train_x[pivot:pivot + batch_size])
                grad = cost.fun_prime(train_y[pivot:pivot+batch_size], forward)
                backward = self._backward(grad)
                pivot = pivot + batch_size
            predictions = self._forward(train_x)
            # error = np.mean(cost.fun(train_y, predictions), 0)
            accuracy = calcAccuracy(predictions, train_y)
            Utils.printProgressBar(i + 1, epochs, suffix = (str(accuracy)+"%"))

