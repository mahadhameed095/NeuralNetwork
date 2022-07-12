import numpy as np
from Layer import Layer
from Utils import getWindows


class Conv2d(Layer):

    def __init__(self, input_shape: tuple, num_kernels: int, kernelSize: int, learning_rate: float) -> None:
        super().__init__()
        stride = 1
        padding = 0
        image_numbers, image_channels, image_height, image_width = input_shape
        output_shape = (image_numbers, num_kernels, int((image_height - kernelSize + 2 * padding) / stride) + 1,
                        int((image_width - kernelSize + 2 * padding) / stride) + 1)
        self._kernels = np.random.randn(num_kernels, image_channels, kernelSize, kernelSize)
        self._bias = np.random.randn(output_shape[1], output_shape[2], output_shape[3])
        self._input = None
        self._alpha = learning_rate

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
        return np.einsum("BCHWKM,NCKM->BNHW", getWindows(input, self._kernels.shape[2]), self._kernels) + self._bias

    def backward(self, outputGradient: np.ndarray) -> np.ndarray:
        kernel_gradient = np.einsum("BCHWKM, BNHW->NCKM", getWindows(self._input, self._kernels.shape[2]),
                                    outputGradient)
        bias_gradient = np.sum(outputGradient, axis=0)
        input_gradient = np.einsum("BNHWKM, NCKM->BCHW",
                                   getWindows(outputGradient, self._kernels.shape[2], self._kernels.shape[2] - 1),
                                   np.rot90(self._kernels, 2, axes=(2, 3)))
        self._kernels = self._kernels - self._alpha * kernel_gradient
        self._bias = self._bias - self._alpha * bias_gradient
        return input_gradient

    # def __init__(self, input_shape : tuple, num_kernels : int, kernelDim : int, learning_rate : float) -> None:
    #     super().__init__()
    #     input_depth, input_width, input_height = input_shape
    #     self._trainable = False
    #     self._kernels = np.random.randn(num_kernels, input_depth, kernelDim, kernelDim)
    #     self._bias = np.random.randn(num_kernels, input_width - kernelDim + 1, input_height - kernelDim + 1)
    #     self._input = None
    #     self._alpha = learning_rate
    #
    # # def forward(self, input: np.ndarray) -> np.ndarray:
    # #     self._input = input
    # #     output = np.empty_like(self._bias)
    # #     for i in range(input.shape[0]):
    # #         output[i] = correlate(input, self._kernels[i], 'valid')
    # #     return output + self._bias
    # #
    # # def backward(self, outputGradient: np.ndarray) -> np.ndarray:
    # #     kernelGradients = np.empty_like(self._kernels)
    # #     inputGradients = np.empty_like(self._input)
    # #     for i in range(self._kernels.shape[0]):
    # #         for j in range(self._kernels.shape[1]):
    # #             kernelGradients[i, j] = correlate2d(self._input[j], outputGradient[i], 'valid')
    # #             inputGradients[j] += convolve2d(outputGradient[i], self._kernels[i, j], 'full')
    # #     self._bias = self._bias - self._alpha * outputGradient
    # #     self._kernels = self._kernels - self._alpha * kernelGradients
    # #     return inputGradients
