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
from warnings import warn
class Network():
    def __init__(self, input_shape : tuple, learning_rate : float = None) -> None:
        self._layers : list = []
        self._args : dict = {"input_shape" : input_shape, "learning_rate" : learning_rate}

    @classmethod
    def FromFile(cls, path) -> 'Network':
        instance = cls(())
        instance.load(path)
        return instance

    def dense(self, num_neurons : int, learning_rate : float = None) -> 'Network':
        assert num_neurons > 0, "The number of neurons can not be zero or negative."
        instance = Dense()
        self._args = instance._initLayer(num_neurons, self._args, learning_rate)
        self._layers.append(instance)
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
        instance = Conv2d()
        self._args = instance._initLayer(num_kernels, kernel_size, self._args, learning_rate)
        self._layers.append(instance)
        return self

    def flatten(self):
        instance = Flatten()
        self._args = instance._initLayer(self._args)
        self._layers.append(instance)
        return self

    def predict(self, input : np.ndarray) -> np.ndarray:
        for layer in self._layers:
            input = layer.forward(input)
        return input
    
    def save(self, path : str) -> None:
        to_save = dict()
        offset = 0
        model = []
        for layer in self._layers:
            if layer._trainable:
                to_save['W'+str(offset)] = layer._weights
                to_save['B'+str(offset)] = layer._bias
                model.append([type(layer).__name__, layer._alpha])
                offset = offset + 1 
            else:
                model.append([type(layer).__name__, None])
        model.append([self._args, None])
        to_save["model_configuration"] = np.array(model, dtype=object)
        np.savez(path, **to_save)
    
    def load(self, path : str) -> None:
        if len(self._layers) > 0:
            warn("The network was not empty. Overwriting the network.")
        self._layers = []
        data = np.load(path, allow_pickle=True)
        model = data["model_configuration"].tolist()
        offset = 0
        self._args = model.pop()[0]
        for layer in model:
            instance =  globals()[layer[0]]()
            if instance._trainable:
                instance._weights = data['W'+str(offset)]
                instance._bias = data['B'+str(offset)]
                instance._alpha = layer[1]
                offset = offset + 1
            self._layers.append(instance) 

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