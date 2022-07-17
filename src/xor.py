import numpy as np
# from Network import Network
# from Trainer import Trainer
# from Conv2d import Conv2d
import time
from Flatten import Flatten
from Softmax import Softmax
from time import process_time
def getWindows1(input: np.ndarray, kernel_size: int, padding: int = 0, stride: int = 1, dilate: bool = False):
    dim = int(np.sqrt(input.shape[2]))
    working_input = input.reshape(input.shape[0], input.shape[1], dim, dim)
    working_pad = padding
    # dilate the input if necessary
    if dilate:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)),
                               mode='constant',
                               constant_values=(0.,))
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides
    out_size =  working_input.shape[2] - kernel_size + 1
    return (
        np.lib.stride_tricks.as_strided(
            working_input,
            (working_input.shape[0], working_input.shape[1], out_size, out_size, kernel_size, kernel_size),
            (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
        )
        .reshape(
            working_input.shape[0], 
            working_input.shape[1], 
            out_size * out_size, 
            kernel_size * kernel_size
        )
        .swapaxes(2, 3)
        .reshape(
            working_input.shape[0], 
            working_input.shape[1] * kernel_size * kernel_size,
            out_size * out_size
        )
    )

def getWindows2(input: np.ndarray, kernel_size: int, padding: int = 0, stride: int = 1, dilate: bool = False):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)),
                               mode='constant',
                               constant_values=(0.,))
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides
    return np.lib.stride_tricks.as_strided(
        working_input,
        (working_input.shape[0], working_input.shape[1], working_input.shape[2] - kernel_size + 1, working_input.shape[3] - kernel_size + 1,
         kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )

# x = np.arange(1, 10).reshape(3, 3)
# print(x.reshape(-1))
# print(np.pad(x, ((1,), (1,)), mode='constant', constant_values=(0.,)).reshape(-1))

batch_size = 100
kernel_size = 10
channels = 3
dim = 200
out_size = dim - kernel_size + 1

# x = np.arange(1, 33).reshape(1, 2, 16)
# print(x)
# print(
#     np.lib.stride_tricks.as_strided(x, (1, 2, 3, 3, 2, 2), (128, 64, 16, 4, 16, 4)).reshape(1, 2, 9, 4).swapaxes(2, 3)
# )
x = np.arange(1, batch_size * channels * dim * dim + 1).reshape(batch_size, channels, dim * dim).astype(np.float32)
# print(getWindows1(x, kernel_size))
y = np.arange(1, batch_size * channels * dim * dim + 1).reshape(batch_size, channels, dim, dim).astype(np.float32)
# print(y.shape)
# print(x)
# start = process_time()
# windows = getWindows1(x, kernel_size)
# print("Time taken ->",process_time() - start)

start = process_time()
windows = getWindows2(y, kernel_size)
print("Time taken ->",process_time() - start)
# [[[[ 1  2  3  4]
#    [ 5  6  7  8]
#    [ 9 10 11 12]
#    [13 14 15 16]]

#   [[17 18 19 20]
#    [21 22 23 24]
#    [25 26 27 28]
#    [29 30 31 32]]]]




# strides = y.strides + y.strides
# print(strides)
# z = as_strided(x, shape=(3, 3, 2, 2), strides=strides)

# w = as_strided(y, shape=(4, 9), strides=(16, 8))
# from MeanSquaredError import MeanSquaredError


# x = np.array([[0, 0], 
#               [0, 1],
#               [1, 0],
#               [1, 1]], dtype="float32")

# y = np.array([[0], 
#               [1], 
#               [1], 
#               [0]], dtype = "float32")

# net = Network()
# net.dense(2, 3, 0.2).tanh().dense(3, 1, 0.2).tanh()
# trainer = Trainer(net)
# trainer.train(100, x, y, MeanSquaredError())

# print(net.predict(x))
