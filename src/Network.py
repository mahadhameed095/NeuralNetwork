import numpy as np
from Dense import Dense
from Tanh import Tanh
from Sigmoid import Sigmoid
from Softmax import Softmax
from pickle import dump, load
class Network():
    def __init__(self, inputSize) -> None:
        self._layers = []
        self._prevNumNeurons = inputSize

    def dense(self, num_neurons : int, learning_rate : float) -> 'Network':
        assert num_neurons > 0, "The number of neurons can not be zero or negative."
        self._layers.append(Dense(self._prevNumNeurons, num_neurons, learning_rate))
        self._prevNumNeurons = num_neurons
        return self
    
    def tanh(self) -> 'Network':
        self._layers.append(Tanh())
        return self

    def sigmoid(self) -> 'Network':
        self._layers.append(Sigmoid())
        return self
    def softmax(self) -> 'Network':
        self._layers.append(Softmax())
        return self

    def predict(self, input : np.ndarray) -> np.ndarray:
        for layer in self._layers:
            input = layer.forward(input)
        return input
    
    def save(self, path : str) -> None:
        with open(path, "wb") as f:
            dump(self._layers, f)
    
    def load(self, path : str) -> None:
        with open(path, "rb") as f:
            self._layers = load(f)