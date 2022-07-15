import numpy as np
from Dense import Dense
from Relu import Relu
from Tanh import Tanh
from Sigmoid import Sigmoid
from Softmax import Softmax
from Conv2d import Conv2d
from Flatten import Flatten
from pickle import dump, load
class Network():
    def __init__(self) -> None:
        self._layers = []
        self.init = False

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
        self._layers.append(self._prevNumNeurons)
        with open(path, "wb") as f:
            dump(self._layers, f)
    
    def load(self, path : str) -> None:
        with open(path, "rb") as f:
            self._layers = load(f)
        self._prevNumNeurons = self._layers.pop()