import numpy as np
# from Network import Network
# from Trainer import Trainer
from Conv2d import Conv2d
import time
from Flatten import Flatten
from Softmax import Softmax
from time import process_time_ns
from Utils import im2Col
from numpy.lib.stride_tricks import as_strided

# batch_size = 256
# kernel_size = 3
# channels = 64
# dim = 50
# stride = 1
# out_size = int((dim - kernel_size)/stride + 1)
# x = np.arange(1, batch_size * channels * dim * dim + 1).reshape(batch_size, channels, dim * dim).astype(np.float32)
# # print(x.reshape(batch_size, channels, dim, dim))
# start = process_time_ns()
# im2Col(x, kernel_size, stride=stride)
# end = process_time_ns()
# print((end - start)/1e9)
# view[0, 0:5] = 0
# view = as_strided(kernels, kernels.shape, strides=tuple(reversed(kernels.strides)))
# kernels[0, 5:10] = 0
# print(flipped)

# kernels[0, 5:20] = 0
# print(kernels)
# print(flipped) 
batch = 1
channels = 2
k_size = 2
dim = 4
num = 3
o_size = dim - k_size + 1
np.random.seed(1)

input = np.arange(1, batch * channels * dim * dim + 1).reshape(batch, channels, dim * dim).astype(np.float32)
print(input)
convie = Conv2d()
convie._initLayer(num, k_size, {"input_shape" : input.shape}, learning_rate=0.1)
forward = convie.forward(input)
back = convie.backward(forward)
print(forward)
print(back)

# input = np.array(np.random.randint(1, 5, (3, 12)), order='F')
# input[:, 0:4] = 0
# input[:, 4:8] = 1
# input[:, 8:12] = 2
# view = np.reshape(input.ravel(order='F'), (3, 12), 'F')
# input[0, 0] = 5
# print(view)
# print(view.flags)
# print(input)
# print(input.flags)
# convie._weights = np.random.randint(1, 5, (num, channels * k_size * k_size))
# start = process_time_ns()
# out = convie.forward(inputArr.reshape(batch, channels, dim * dim))
# end = process_time_ns()
# print(end - start)
# out = out.reshape(batch, num, o_size, o_size)


# for i in range(batch):
#     for j in range(num):
#         other = correlate(inputArr[i], convie._weights.reshape(num, channels, k_size, k_size)[j], mode='valid')
#         print(np.array_equal(out[i, j], other.squeeze()))
# print(kernels)
# print(out_mat)

# print(np.einsum("ND, BDO -> BNO", kernels, out_mat))

# print(kernels @ out_mat[0])
# print(kernels @ out_mat[1])
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


