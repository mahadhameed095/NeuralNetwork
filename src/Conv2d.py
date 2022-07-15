import numpy as np
from Layer import Layer
from Utils import getWindows


class Conv2d(Layer):

    def __init__(self, num_kernels: int, kernelSize: int, learning_rate: float = None) -> None:
        super().__init__()
        self._trainable = True
        self._kernels = None
        self._bias = None
        self._input = None
        self._alpha = learning_rate
        self._cache = (num_kernels, kernelSize)

    def _initLayer(self, argsDict: dict) -> dict:
        assert "input_shape" in argsDict, "Input Shape not specified"
        num_kernels = self._cache[0]
        kernelSize = self._cache[1]
        stride = 1
        padding = 0
        image_numbers, image_channels, image_height, image_width = argsDict["input_shape"]
        output_shape = (image_numbers, num_kernels, int((image_height - kernelSize + 2 * padding) / stride) + 1,
                        int((image_width - kernelSize + 2 * padding) / stride) + 1)
        self._kernels = np.random.randn(num_kernels, image_channels, kernelSize, kernelSize) / 1000000
        self._bias = np.random.randn(output_shape[1], output_shape[2], output_shape[3]) / 1000000
        if self._alpha == None:
            assert "learning_rate" in argsDict, "Learning Rate not specified. Either specify in the layer constructor, or pass as argument in dictionary"
            self._alpha = argsDict["learning_rate"]
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
        self._input = input
        self._cache = getWindows(input, self._kernels.shape[2])
        # Cached so it can be reused in backward method
        return np.einsum("BCHWKM,NCKM->BNHW", self._cache, self._kernels) + self._bias

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        kernel_gradient = np.einsum("BCHWKM, BNHW->NCKM", self._cache,
                                    outputGradient)
        bias_gradient = np.sum(outputGradient, axis = 0)
        input_gradient = np.einsum("BNHWKM, NCKM->BCHW",
                                   getWindows(outputGradient, self._kernels.shape[2], self._kernels.shape[2] - 1),
                                   np.rot90(self._kernels, 2, axes=(2, 3)))
        self._kernels = self._kernels - self._alpha * kernel_gradient
        self._bias = self._bias - self._alpha * bias_gradient
        return input_gradient