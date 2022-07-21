from functools import cache
import numpy as np
from Layer import Layer
from Utils import im2Col

class Conv2d(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._trainable = True
        self._weights = None
        self._bias = None
        self._alpha = None
        self._cache = None
        self._kernelShape = None
        self._kernelView = None

    def _initLayer(self, num_kernels: int, kernelSize: int, argsDict: dict, learning_rate: float = None) -> dict:
        stride = 1
        padding = 0
        image_numbers, image_channels, image_size= argsDict["input_shape"]
        image_dim = np.sqrt(image_size)
        output_shape = (image_numbers, num_kernels, (int((image_dim - kernelSize + 2 * padding) / stride) + 1)**2)
        self._kernelShape = (num_kernels, image_channels, kernelSize, kernelSize)
        self._weights = np.random.randn(num_kernels, kernelSize * kernelSize * image_channels) * 1e-3
        self._kernelView = np.reshape(self._weights.ravel(), (image_channels, kernelSize * kernelSize * num_kernels), order='F')
        self._bias = np.random.randn(output_shape[1], output_shape[2]) * 1e-3
        if learning_rate == None:
            assert argsDict["learning_rate"] is not None, "Learning Rate not specified. Either specify in the layer constructor, or pass as argument in the network constructor"
            self._alpha = argsDict["learning_rate"]
        else:
            self._alpha = learning_rate

        argsDict["input_shape"] = output_shape
        return argsDict

    def forward(self, input: np.ndarray) -> np.ndarray:
        self._cache = im2Col(input, self._kernelShape[2])
        # Cached so it can be reused in backward method
        return np.einsum("ND, BDO -> BNO", self._weights, self._cache) + self._bias

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        kernel_gradient = np.einsum("BNO, BOD ->ND", outputGradient, self._cache.swapaxes(1, 2))
        bias_gradient = np.sum(outputGradient, axis = 0)

        num_kernels, channels, kernel_size, _ =self._kernelShape

        rotatedKernel = np.flip(np.split(self._weights, channels, axis=1), axis = 2).reshape(channels, kernel_size**2 * num_kernels)

        input_gradient = np.einsum("CD, BDO->BCO",rotatedKernel, im2Col(outputGradient, kernel_size, kernel_size - 1))

        self._weights = self._weights - self._alpha * kernel_gradient
        self._bias = self._bias - self._alpha * bias_gradient
        return input_gradient



"""
    This link was a huge help.
        https://blog.ca.meron.dev/Vectorized-CNN/
"""