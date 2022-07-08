import numpy as np
from Dense import Dense
from Tanh import Tanh
from Sigmoid import Sigmoid
class Network():
    def __init__(self) -> None:
        self._layers = []

    def dense(self, inputSize, outputSize, learning_rate) -> 'Network':
        self._layers.append(Dense(inputSize, outputSize, learning_rate))
        return self
    
    def tanh(self) -> 'Network':
        self._layers.append(Tanh())
        return self

    def sigmoid(self) -> 'Network':
        self._layers.append(Sigmoid())
        return self


    def predict(self, input) -> np.array:
        for layer in self._layers:
            input = layer.forward(input)
        return input