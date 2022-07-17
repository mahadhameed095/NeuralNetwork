import numpy as np
from Layer import Layer
from Utils import getWindows

class Conv2d(Layer):

    def __init__(self) -> None:
        super().__init__()
        self._trainable = True
        self._weights = None
        self._bias = None
        self._alpha = None
        self._cache = [None, None]

    def _initLayer(self, num_kernels: int, kernelSize: int, argsDict: dict, learning_rate: float = None) -> dict:
        stride = 1
        padding = 0
        image_numbers, image_channels, image_height, image_width = argsDict["input_shape"]
        output_shape = (image_numbers, num_kernels, int((image_height - kernelSize + 2 * padding) / stride) + 1,
                        int((image_width - kernelSize + 2 * padding) / stride) + 1)

        self._weights = np.random.randn(num_kernels, kernelSize * kernelSize * image_channels) * 1e-3             
        self._bias = np.random.randn(output_shape[1], output_shape[2] * output_shape[3]) * 1e-3
        if learning_rate == None:
            assert argsDict["learning_rate"] is not None, "Learning Rate not specified. Either specify in the layer constructor, or pass as argument in the network constructor"
            self._alpha = argsDict["learning_rate"]
        else:
            self._alpha = learning_rate

        self._cache[1] = np.rot90(self._weights, 2, axes=(2, 3))
        argsDict["input_shape"] = output_shape
        return argsDict

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
            Explaining how np.einsum works:
         B -> Number of Samples in input
         C -> Number of channels in each sample
         H -> The height of the output of this function
         W -> The width of the output of this function
         K -> The height of the kernel
         M -> The width of the kernel
         N -> The number of filters/Kernels/(Channels in output)

         The windows are of (B, C, H, W, K, M) shape and the kernels are of (N, C, K, M) shape

         Explaining the einsum.
            The K and M line up, as they should. This will compute the element wise product of those dimensions

        for B in Batches:
            for N in num_filers:
                for H in O_H:
                    for W in O_W:
                        for C in Channels:
                            for K in Kernel_height:
                                form M in Kernel_width:
                                    out[B, N, H, W] += window[B, C, H, W, K, M] * kernel[N, C, K, M]
                                    # (BNHW <- BCHWKM, NCKM)Einsum Equation
        """
        self._cache[0]= getWindows(input, self._weights.shape[2])
        # Cached so it can be reused in backward method
        return np.einsum("BCHWKM,NCKM->BNHW", self._cache[0], self._weights) + self._bias

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        kernel_gradient = np.einsum("BCHWKM, BNHW->NCKM", self._cache[0],
                                    outputGradient)
        bias_gradient = np.sum(outputGradient, axis = 0)
        input_gradient = np.einsum("BNHWKM, NCKM->BCHW",
                                   getWindows(outputGradient, self._weights.shape[2], self._weights.shape[2] - 1),
                                   self._cache[1])
        self._weights = self._weights - self._alpha * kernel_gradient
        self._bias = self._bias - self._alpha * bias_gradient
        return input_gradient



"""
    This link was a huge help.
        https://blog.ca.meron.dev/Vectorized-CNN/
"""