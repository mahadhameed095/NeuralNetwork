import numpy as np
# from Network import Network
# from Trainer import Trainer
# from Conv2d import Conv2d
import time
from Utils import getWindows
from Flatten import Flatten
from h5py import File
from pickle import dump, load

np.random.seed(1)
data = np.random.randn(2000, 100, 1000)
# np.save("2tea", data)

with open("data", "wb") as f:
    dump(data, f)


# with File("DataSet", 'wb') as f:
#     f.create_dataset("weights", data=data)

# from scipy.signal import correlate
# np.random.seed(1)
# x = np.random.randint(1, 5, (3, 2, 4, 4)).astype(np.float32)
#
# c = Conv2d(x.shape, 2, 2, 0.1)
#
# f = c.forward(x)
#
# out = correlate(x[0], c._kernels[0], 'valid')
# out = np.squeeze(out, axis = 0).astype(np.float32)
# print(out == f[0][0].astype(np.float32))
#
# print(f[0][0])
# print(out)


# k = np.random.randint(0, 2, (2, 3, 3))
# print(x)
# print()
# print(k)
# print()
# print(correlate(x, k, 'valid').reshape(3, 3))

# from MeanSquaredError import MeanSquaredError

# x = np.array([[0, 0, 0, 0], 
#              [0, 1, 0, 1]], dtype = "float32")
# y = np.array([[0, 1, 1, 0]], dtype = "float32")

# net = Network()
# net.dense(2, 3, 0.2).tanh().dense(3, 1, 0.2).tanh()
# trainer = Trainer(net)
# trainer.train(100, x, y, MeanSquaredError())

# print(net.predict(x))
