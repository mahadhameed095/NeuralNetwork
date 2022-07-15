import numpy as np
from Dense import Dense
from Relu import Relu
from Tanh import Tanh
from Sigmoid import Sigmoid
from Softmax import Softmax
from Conv2d import Conv2d
from Flatten import Flatten
from pickle import dump, load
from math import prod
class Network():
    def __init__(self) -> None:
        self._layers : list = []
        self.init : bool = False

    def dense(self, num_neurons : int, learning_rate : float = None) -> 'Network':
        assert num_neurons > 0, "The number of neurons can not be zero or negative."
        self._layers.append(Dense(num_neurons, learning_rate))
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

    def relu(self) -> 'Network':
        self._layers.append(Relu())
        return self

    def conv2d(self, num_kernels: int, kernel_size: int, learning_rate: float = None) -> 'Network':
        assert num_kernels > 0, "The number of kernels can not be zero or negative."
        assert kernel_size > 0, "The size of kernel can not be zero or negative."
        self._layers.append(Conv2d(num_kernels, kernel_size, learning_rate))
        return self

    def flatten(self):
        self._layers.append(Flatten())
        return self

    def predict(self, input : np.ndarray) -> np.ndarray:
        for layer in self._layers:
            input = layer.forward(input)
        return input
    
    def save(self, path : str) -> None:
        self._layers.append(self.init)
        with open(path, "wb") as f:
            dump(self._layers, f)

    
    def load(self, path : str) -> None:
        with open(path, "rb") as f:
            self._layers = load(f)
        self.init = self._layers.pop()
    def print_summary(self) -> None:
        sum = 0
        toPrint = ""
        width = 60
        print('-' * width)
        for layer in self._layers:
            toPrint = "| " + str(type(layer).__name__) 
            if layer._trainable:
                toPrint += " -> W" + str(layer._weights.shape) + ", B" + str(layer._bias.shape)
                sum += prod(layer._weights.shape) + prod(layer._bias.shape)
            toPrint += ' ' * (width - len(toPrint)) + '|'
            print(toPrint)
        print('|' + '-' * (width - 1) + '|')
        toPrint = "| Total number of parameters in the network = " + str(sum) 
        toPrint += ' ' * (width - len(toPrint)) + '|'
        print(toPrint)
        print('-' * width)